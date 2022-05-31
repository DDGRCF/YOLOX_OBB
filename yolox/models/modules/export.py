import torch
import torch.nn as nn
import torch.nn.functional as F

def AdaptiveAvgPool2d__forward(ctx, x, align_corners=False):
    size = x.shape[2:]
    kernel = [int(size[i] / ctx.output_size[i]) for i in range(0, len(size))]
    x = F.avg_pool2d(x, kernel_size=kernel, stride=kernel, padding=0, ceil_mode=False)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)
    return x
