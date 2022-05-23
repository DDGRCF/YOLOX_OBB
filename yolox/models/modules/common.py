import warnings
import torch
import torch.nn as nn
import numpy as np
import math
import timm

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Scale(torch.nn.Module):
    def __init__(self, value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(value, dtype=torch.float32))
    
    def forward(self, input):
        return self.scale * input

class ConvN(nn.Module):
    def __init__(
        self, 
        c1, 
        c2, 
        k=1, s=1, 
        p=-1,g=1, 
        b=True,
        norm_func=None, 
        act_func=None,
        init_func=None,
        inplace=False):
        super().__init__()
        if isinstance(p, int):
            if p == -1:
                p = autopad(k)
            elif p < -1:
                raise ValueError
        else:
            if p == None:
                p = autopad(k)

        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=b)
        self.bn = norm_func(c2) if norm_func is not None else nn.Identity()
        self.act = act_func(inplace=inplace) if act_func is not None else nn.Identity() 
        if init_func is not None:
            init_func(self.conv)

    def forward(self, x):
        x = self.conv(x)
        if self.bn == nn.LayerNorm:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.bn(x)
        x = self.act(x)
        if self.bn == nn.LayerNorm:
            x = x.permute(0, 3, 1, 2)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Conv(nn.Module):
    def __init__(
        self, 
        c1, 
        c2, 
        k=1, s=1, 
        p=-1,g=1, 
        b=False,
        norm_func=nn.BatchNorm2d, 
        act_func=nn.SiLU,
        init_func=None,
        inplace=False):
        super().__init__()
        if isinstance(p, int):
            if p == -1:
                p = autopad(k)
            elif p < -1:
                raise ValueError
        else:
            if p == None:
                p = autopad(k)

        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=b)
        self.bn = norm_func(c2) if norm_func is not None else nn.Identity()
        self.act = act_func(inplace=inplace) if act_func is not None else nn.Identity() 
        if init_func is not None:
            init_func(self.conv)

    def forward(self, x):
        x = self.conv(x)
        if self.bn == nn.LayerNorm:
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.bn(x)
        x = self.act(x)
        if self.bn == nn.LayerNorm:
            x = x.permute(0, 3, 1, 2)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    def __init__(
        self, 
        c1,
        c2,
        ksize, stride, 
        pad=-1, 
        bias=False,
        norm_func=nn.BatchNorm2d, 
        act_func=nn.SiLU,
        init_func=None):
        super().__init__(
            c1,
            c2,
            ksize, stride, 
            pad, g=math.gcd(c1, c2),
            bias=bias,
            norm_func=norm_func,
            act_func=act_func,
            init_fucn=init_func
        )


class Linear(nn.Module):
    def __init__(
        self, 
        c1, 
        c2, 
        b=False,
        norm_func=nn.LayerNorm, 
        act_func=nn.GELU,
        init_func=None):
        super().__init__()
        self.linear = nn.Linear(c1, c2, b)
        self.bn = norm_func(c2) if \
            (norm_func is not None and isinstance(norm_func, nn.Module)) else nn.Identity()
        self.act = act_func() if \
            (act_func is not None and isinstance(norm_func, nn.Module)) else nn.Identity() 
        if init_func is not None:
            init_func(self.linear)

    def forward(self, x):   
        return self.act(self.bn(self.linear(x)))

    def fuseforward(self, x):
        return self.act(self.linear(x))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=-1, g=1, act=True, **kwargs):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act, **kwargs)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True, **kwargs):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act, **kwargs)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act, **kwargs)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13), **kwargs):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, **kwargs)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1, **kwargs)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))