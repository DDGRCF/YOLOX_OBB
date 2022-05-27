import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAvgPool2d_E(nn.AdaptiveAvgPool2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.align_corners = kwargs.pop("align_corners", False)
    
    def forward(self, x):
        size = x.shape[2:]
        kernel = [int(size[i] / self.output_size[i]) for i in range(0, len(size))]
        x = F.avg_pool2d(x, kernel_size=kernel, stride=kernel, padding=0, ceil_mode=False)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=self.align_corners)
        return x