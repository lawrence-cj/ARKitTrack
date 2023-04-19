import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils_rgbd as prutils
import torch.nn.functional as F
import cv2
import numpy as np


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""

    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None,
                 joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search': transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class BEVDSupProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_depths'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_depths'])
            data['search_images'], data['search_anno'], data['search_depths'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_depths'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, depth_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                               data[s + '_anno'],
                                                                               self.search_area_factor[s],
                                                                               self.output_sz[s],
                                                                               masks=data[s + '_depths'])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_depths'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=depth_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        if isinstance(data["template_depths"], list):
            data["template_depths"] = torch.Tensor(data["template_depths"][0]).unsqueeze(0)
        if isinstance(data["search_depths"], list):
            data["search_depths"] = torch.Tensor(data["search_depths"][0]).unsqueeze(0)

        data['template_masks'] = data["template_depths"].clone().detach() * 0
        data['search_masks'] = data["search_depths"].clone().detach() * 0

        return data


def pixel2point_cloud(w, h, color, depth):
    """

    :param w:
    :param h:
    :param color: (N, 3, H, W)  RGB
    :param depth: (N, 1, H, W)
    :param K: (N, 3, 3)
    :param T: world-to-camera (N, 4, 4)
    :return:
    """
    xs, ys = np.meshgrid(range(w), range(h), indexing='xy')
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    ones = np.ones_like(ys).astype(np.float32)
    coords = np.stack((xs, ys, ones), axis=0)  # (3, H, W)

    ns = depth.shape[0]
    coords = torch.from_numpy(coords).unsqueeze(0).flatten(2)  # (1, 3, H*W)
    coords = coords.repeat(ns, 1, 1)  # (N, 3, H*W)
    coords = coords.to(depth.device)

    coords[:, 2, :] = coords[:, 2, :] * depth.flatten(2)

    cloud = torch.cat((coords, color.flatten(2)), dim=1)  # (N, 6, L)

    return cloud


def point_cloud2BEV(w, h, max_depth, cloud, depth_map, box=None, downsample=1):
    """
    :param cloud:  (1, 6, H*W)
    :param box:  (4,) x,y,w,h
    :param depth_map: array
    :param max_depth
    """
    cloud = cloud[..., ::downsample]

    coords = cloud[:, :3]  # (N, 3, L) [x, y, z]
    bgr = cloud[:, 3:]  # (N, 3, L) [r, g, b]

    if box is not None:
        box[2:] += box[:2]
        box[0::2] = np.clip(box[0::2], 0, w - 1)
        box[1::2] = np.clip(box[1::2], 0, h - 1)
        box = box.astype(int)
        x1, y1, x2, y2 = box.astype(int)

        object_depth = depth_map[y1:y2, x1:x2]
        center_depth = np.mean(object_depth)
        near_depth = center_depth - 20
        far_depth = center_depth + 20

        bev_box = np.array([near_depth, x1, far_depth, x2]).astype(int)
        bev_box[0::2] = np.clip(bev_box[0::2], 0, max_depth - 1)
        bev_box[1::2] = np.clip(bev_box[1::2], 0, w - 1)
    else:
        bev_box = None

    # BEV coords (x, y, z) -> (z, x, y)
    coords = coords[:, [2, 0, 1], ...]

    ns = coords.shape[0]
    assert ns == 1, "only support batch size = 1"

    bev_r = torch.zeros((w, max_depth))
    bev_r.index_put_((coords[0, 1].long(), coords[0, 0].long()), bgr[0, 0], accumulate=True)

    bev_g = torch.zeros((w, max_depth))
    bev_g.index_put_((coords[0, 1].long(), coords[0, 0].long()), bgr[0, 1], accumulate=True)

    bev_b = torch.zeros((w, max_depth))
    bev_b.index_put_((coords[0, 1].long(), coords[0, 0].long()), bgr[0, 2], accumulate=True)

    bev_map_torch = torch.stack((bev_r, bev_g, bev_b), dim=0)

    bev_map_arr = bev_map_torch.permute(1, 2, 0).numpy().astype(np.uint8)

    if bev_box is not None:
        bev_map_arr = cv2.rectangle(bev_map_arr.copy(), bev_box[:2], bev_box[2:], (0, 255, 0), 2)

    return bev_map_torch, bev_map_arr


def point_cloud2SiV(w, h, max_depth, cloud, depth_map, box=None, downsample=1):
    """
    :param cloud:  (1, 6, H*W)
    :param box:  (4,) x,y,w,h
    :param depth_map: array
    :param max_depth
    """
    cloud = cloud[..., ::downsample]

    coords = cloud[:, :3]  # (N, 3, L) [x, y, z]
    bgr = cloud[:, 3:]  # (N, 3, L) [r, g, b]

    if box is not None:
        box[2:] += box[:2]
        box[0::2] = np.clip(box[0::2], 0, w - 1)
        box[1::2] = np.clip(box[1::2], 0, h - 1)
        box = box.astype(int)
        x1, y1, x2, y2 = box.astype(int)

        object_depth = depth_map[y1:y2, x1:x2]
        center_depth = np.mean(object_depth)
        near_depth = center_depth - 20
        far_depth = center_depth + 20

        bev_box = np.array([near_depth, y1, far_depth, y2]).astype(int)
        bev_box[0::2] = np.clip(bev_box[0::2], 0, max_depth - 1)
        bev_box[1::2] = np.clip(bev_box[1::2], 0, h - 1)
    else:
        bev_box = None

    # BEV coords (x, y, z) -> (z, y, x)
    coords = coords[:, [2, 1, 0], ...]

    ns = coords.shape[0]
    assert ns == 1, "only support batch size = 1"

    bev_r = torch.zeros((h, max_depth))
    bev_r.index_put_((coords[0, 1].long(), coords[0, 0].long()), bgr[0, 0], accumulate=True)

    bev_g = torch.zeros((h, max_depth))
    bev_g.index_put_((coords[0, 1].long(), coords[0, 0].long()), bgr[0, 1], accumulate=True)

    bev_b = torch.zeros((h, max_depth))
    bev_b.index_put_((coords[0, 1].long(), coords[0, 0].long()), bgr[0, 2], accumulate=True)

    bev_map_torch = torch.stack((bev_r, bev_g, bev_b), dim=0)

    bev_map_arr = bev_map_torch.permute(1, 2, 0).numpy().astype(np.uint8)

    if bev_box is not None:
        bev_map_arr = cv2.rectangle(bev_map_arr.copy(), bev_box[:2], bev_box[2:], (0, 255, 0), 2)

    return bev_map_torch, bev_map_arr
