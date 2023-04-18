import math
from typing import Tuple, List

import torch
import torch.nn as nn

from mmdet3d.ops import bev_pool
import torch.nn.functional as F


class LiftSplatShoot(nn.Module):
    def __init__(self, in_channels, out_channels, bounds):
        super(LiftSplatShoot, self).__init__()

        self.xbound = bounds[0]
        self.ybound = bounds[1]
        self.zbound = bounds[2]
        self.dbound = bounds[3]

        self.image_size = (384, 384)
        self.feature_size = (24, 24)

        dx, bx, nx = self.gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)  # bin_size
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)  # bins

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape

        self.C = out_channels

        # depthsup2
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.use_gauss = True
        if self.use_gauss:
            self.depthnet = nn.Sequential(
                nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, self.C, 1),
            )
        else:
            self.depthnet = nn.Sequential(
                nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, self.D + self.C, 1),
            )

        # # depthsup1 do not use real depth
        # self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)

        # self.camencode = CamEncode(self.D, self.camC, self.downsample)
        # self.bevencode = BevEncode(inC=self.camC, outC=outC)
        #
        # # toggle using QuickCumsum vs. autograd
        # self.use_quickcumsum = True

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor(
            [(row[1] - row[0]) * 10 / (row[2] * 10) for row in [xbound, ybound, zbound]]
        )
        return dx, bx, nx

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape

        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, intrins):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N = intrins.shape[:2]
        assert N == 1, "only support one camera"

        points = self.frustum.unsqueeze(0).unsqueeze(0)  # (1, 1, 118, fH, fW, 3)
        points = points.expand(B, N, -1, -1, -1, -1).unsqueeze(-1)

        # multiply depth
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # (B, N, 1, 1, 1, 3, 1)

        # # inverse(intrins) @
        # # (B, N, 1, 1, 1, 3, 3) @ (B, N, 118, fH, fW, 3, 1) -> (B, N, 118, fH, fW, 3, 1)
        # points = torch.linalg.solve(intrins.view(B, N, 1, 1, 1, 3, 3), points)
        # points = points.squeeze(-1)  # (B, N, 118, fH, fW, 3)

        camera2lidar_rots = torch.Tensor([[1, 0, 0],
                                          [0, 0, 1],
                                          [0, -1, 0]]).reshape(1, 1, 3, 3).to(intrins.device)

        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)  # (B, N, 118, fH, fW, 3)
        # points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        return points

    def bev_pooling(self, geom_feats, x):  # (B, N, D, fh, fw, 3)
        B, N, D, H, W, C = x.shape  # (B, N, D, fh, fw, C)
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)
        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        # filter out points that are outside box
        kept = (
                (geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        x = bev_pool(x.cuda(), geom_feats.cuda(), B, self.nx[2], self.nx[0], self.nx[1])
        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)
        return final

    def encode_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape
        assert N == 1, "only support one camera"

        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)  # (BN, 1, D, fh, fw) * (BN, C, 1, fh, fw) -> (BN, C, D, fh, fw)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)  # (B, N, D, fh, fw, C)
        return x  # (B, N, D, fH, fW, C)

    def encode_cam_feats2(self, x, d):
        B, N, C, fH, fW = x.shape
        assert N == 1, "only support one camera"

        x = x.view(B * N, C, fH, fW)

        _d = self.dtransform(d)
        x = torch.cat([_d, x], dim=1)
        x = self.depthnet(x)  # (B * N, C, fH, fW) -> (B * N, self.D + self.C, fH, fW)

        avgd = F.avg_pool2d(d, 16, 16)  # (B*N, 1, fH, fW)  cm
        mean_d = torch.floor(avgd / 1000 * self.D)
        depth = self.create_gaussian(mean_d).softmax(dim=1)

        x = depth.unsqueeze(1) * x.unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x  # (B, N, D, fH, fW, C)

    def forward(self, x_feat, intrins, depth_x=None):
        geom = self.get_geometry(intrins)

        if not self.use_gauss:
            x = self.encode_cam_feats(x_feat, depth_x)  # (B, N, D, fH, fW, C)
        else:
            x = self.encode_cam_feats2(x_feat, depth_x)  # (B, N, D, fH, fW, C)

        x = self.bev_pooling(geom, x)  # (B, C, self.nx[0], self.nx[1])
        x = x[..., 50:]
        return x

    def create_gaussian(self, mean, sigma=0.5):

        x = torch.arange(0, self.D).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(mean.device)
        x = x.expand(mean.shape[0], self.D, 24, 24)
        y = (1/math.sqrt(2*math.pi)*sigma) * torch.exp(-(x-mean)**2/2*sigma**2)
        return y
