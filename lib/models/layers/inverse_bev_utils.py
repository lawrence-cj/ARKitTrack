import math
from typing import Tuple, List

import torch
import torch.nn as nn
from functools import partial
from mmdet3d.ops import bev_pool
import torch.nn.functional as F
from lib.models.layers.attn_blocks import Block


class InvBEV(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InvBEV, self).__init__()

        self.image_size = (384, 384)
        self.feature_size = (24, 24)
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        # self.att = nn.Sequential(
        #     Block(dim=in_channels, num_heads=8, mlp_ratio=1, qkv_bias=True, drop=0,  # input B,N,C
        #                  attn_drop=0, drop_path=0, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU),
        #     #Block(dim=in_channels, num_heads=8, mlp_ratio=4, qkv_bias=True, drop=0,  # input B,N,C
        #           #attn_drop=0, drop_path=0, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU),
        # )

    def create_frustum(self, depth_map):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        fdepth = F.avg_pool2d(depth_map, 16, 16) / 100  # (B, 1, fH, fW)  cm ->m
        B, _, _, _ = fdepth.shape

        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float, device=fdepth.device).view(1, 1, 1, fW).expand(B, 1, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float, device=fdepth.device).view(1, 1, fH, 1).expand(B, 1, fH, fW)

        frustum = torch.stack((xs, ys, fdepth), -1)
        return frustum

    def get_geometry(self, intrins, depth_map):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N = intrins.shape[:2]
        assert N == 1, "only support one camera"

        points = self.create_frustum(depth_map).unsqueeze(1)  # (B, 1, 1, fH, fW, 3)
        points = points.unsqueeze(-1)  # (B, 1, 1, fH, fW, 3, 1)

        # multiply depth
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # (B, N, 1, fH, fW, 3, 1)

        # # inverse(intrins) @
        # # (B, N, 1, 1, 1, 3, 3) @ (B, N, 118, fH, fW, 3, 1) -> (B, N, 118, fH, fW, 3, 1)
        # points = torch.linalg.solve(intrins.view(B, N, 1, 1, 1, 3, 3), points)
        # points = points.squeeze(-1)  # (B, N, 118, fH, fW, 3)

        camera2lidar_rots = torch.Tensor([[1, 0, 0],
                                          [0, 0, 1],
                                          [0, -1, 0]]).reshape(1, 1, 3, 3).to(intrins.device)

        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)  # (B, N, 1, fH, fW, 3)

        return points.squeeze(1)

    def forward(self, bev_feat, intrins, depth_x):
        bev_feat = self.conv(bev_feat)

        # B, C, H, W = bev_feat.shape
        # bev_feat = bev_feat.reshape(B, C, -1).permute(0, 2, 1)
        # bev_feat = self.att(bev_feat).permute(0, 2, 1).reshape(B, C, H, W)

        geom = self.get_geometry(intrins, depth_x)  # (B, 1, fH, fW, 3)
        idx = geom[..., :2]  # (B, 1, fH, fW, [x, y])
        idx[..., 0] += 5  # off x
        idx = torch.round(idx / 10 * 50)

        idx = idx[..., 0] * 50 + idx[..., 1]  # (B, 1, fH, fW)
        idx = idx.flatten(2)  # (B, 1, fH*fW)
        bev = bev_feat.flatten(2)
        inverse_bev = torch.gather(bev, index=idx.long().repeat(1, bev.shape[1], 1), dim=-1)

        B, C = inverse_bev.shape[:2]
        fH, fW = self.feature_size
        inverse_bev = inverse_bev.reshape(B, C, fH, fW)

        return inverse_bev


if __name__ == '__main__':
    bev_feat = torch.randn(2, 128, 50, 50)

    intrins = torch.Tensor([[952.828 * 0.4948, 0., 192],
                            [0., 952.828 * 0.5253, 192],
                            [0., 0., 1.]]).view(1, 1, 3, 3)
    intrins = intrins.expand(2, 1, 3, 3)

    depth_x = torch.rand(2, 1, 384, 384) * 1000
    a = InvBEV(128, 128)
    x = a(bev_feat, intrins, depth_x)
    print(x.shape)
