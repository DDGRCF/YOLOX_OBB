import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .common import *
class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, **kwargs):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s), **kwargs)
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g, **kwargs)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class PPM(nn.Module):

    def __init__(self, in_channels, out_channels=512, 
                 sizes=(1, 2, 3, 6), norm_func=None, 
                 act_func=nn.ReLU, export=False,
                 **kwargs):
        super().__init__()
        assert out_channels % len(sizes) == 0
        channels = out_channels // len(sizes)
        self.stages = []
        self.sizes = sizes
        self.export=export
        self.act_func = act_func
        self.norm_func = norm_func
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size, **kwargs) for size in sizes]
        )

        self.bottleneck = Conv(
            in_channels + len(sizes) * channels, in_channels, 1, norm_func=norm_func, act_func=act_func, **kwargs
        )

    def _make_stage(self, features, out_features, size, **kwargs):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = Conv(features, out_features, 1, norm_func=self.norm_func, act_func=self.act_func, **kwargs)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(
            h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats] # why
        # out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        out = self.bottleneck(torch.cat(priors, 1))
        return out

class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y

class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5, **kwargs):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, **kwargs)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, **kwargs)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class ConvNextBlock(nn.Module):
    def __init__(
        self, 
        c,
        e=4,
        d=0.,
        layer_scale_init_value=1e-6):
        super().__init__()
        self.dw1 = nn.Conv2d(c, c, kernel_size=7, padding=3, groups=c)
        self.n1 = nn.LayerNorm(c, eps=1e-4)
        self.pw1 = nn.Linear(c, e * c)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(e * c, c)
        self.g = nn.Parameter(layer_scale_init_value * torch.ones((c)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = timm.model.layer.DropPath(d) if d > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.c1(x)
        x = self.l1(x)
        x = self.l2(x)
        if self.g is not None:
            x = self.g * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, channels, num_heads):
        super().__init__()
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads)
        self.fc1 = nn.Linear(channels, channels, bias=False)
        self.fc2 = nn.Linear(channels, channels, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, **kwargs):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, **kwargs)
        self.cv2 = Conv(c1, c_, 1, 1, **kwargs)
        self.cv3 = Conv(2 * c_, c2, 1, **kwargs)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0, **kwargs) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, **kwargs):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, **kwargs)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, **kwargs)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Coordinates(nn.Module):
    def __init__(self, mode="absolute"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == "absolute":
            h, w = x.size()[-2:]
            # y_loc = torch.linspace(-1, 1, h, device=x.device)
            # x_loc = torch.linspace(-1, 1, w, device=x.device)

            y_loc = torch.arange(-1, 1, 2 / h, device=x.device) # TODO: figout 
            x_loc = torch.arange(-1, 1, 2 / w, device=x.device)
            y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
            y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
            x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
            locations = torch.cat([x_loc, y_loc], 1).to(x)
            x = torch.cat([locations, x], dim=1)
        elif self.mode == "relative":
            raise NotImplementedError
        return x
 
class InstConv(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, k=[3], s=[1], p=[-1], b=[True], a=[nn.ReLU], norm_func=None, **kwargs):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.convs = nn.ModuleList()
        kwargs.pop("p", None)
        kwargs.pop("k", None)
        kwargs.pop("s", None)
        kwargs.pop("b", None)
        kwargs.pop("norm_func", None)
        kwargs.pop("act_func", None)
        a = [a] * n if not isinstance(a, (list, tuple)) else a
        s = [s] * n if not isinstance(s, (list, tuple)) else s
        k = [k] * n if not isinstance(k, (list, tuple)) else k
        p = [p] * n if not isinstance(p, (list, tuple)) else p
        b = [b] * n if not isinstance(p, (list, tuple)) else b
        for i in range(n):
            a_i = a[i] if i < len(a) else None
            b_i = b[i] if i < len(b) else True
            s_i = s[i] if i < len(s) else 1
            k_i = k[i] if i < len(k) else 1
            p_i = p[i] if i < len(p) else -1
            self.convs.append(Conv(c1, c2, k=k_i, s=s_i, p=p_i, b=b_i, act_func=a_i, norm_func=norm_func, **kwargs))
            c1 = c2

    def forward(self, x):
        return nn.Sequential(*self.convs)(x)