import torch
import torch.nn as nn
from yolox.utils.mask_utils import aligned_bilinear
from .common import Conv, DWConv

class CondInstMaskBranch(nn.Module):
    def __init__(
        self,
        in_channels=(256, 512, 1024),
        feat_channel=128,
        out_channel=8,
        num_convs=4,
        depthwise=False,
        out_stride=8,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_convs = num_convs
        self.out_stride = out_stride
        conv = DWConv if depthwise else Conv
        self.refine_modules = nn.ModuleList()
        for c in in_channels:
            self.refine_modules.append(
                conv(c, feat_channel, 3, 1, **kwargs)
            )
        mask_modules = nn.ModuleList() 
        for _ in range(num_convs):
            mask_modules.append(
                conv(feat_channel, feat_channel, 3, 1, **kwargs)
            )
        mask_modules.append(
            conv(feat_channel, out_channel, 3, 1, **kwargs)
        )
        self.add_module('mask_modules', nn.Sequential(*mask_modules))

    def forward(self, features):
        for i, f in enumerate(features):
            if i == 0:
                x = self.refine_modules[i](f)
            else:
                x_p = self.refine_modules[i](f) 

                target_h, target_w = x.shape[2:]
                h, w = x_p.shape[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p

        return self.mask_modules(x)