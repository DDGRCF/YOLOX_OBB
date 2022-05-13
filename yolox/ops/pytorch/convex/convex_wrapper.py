import torch
from torch.autograd import Function
from . import convex_ext


class ConvexSortFunction(Function):

    @staticmethod
    def forward(ctx, pts, masks, circular):
        idx = convex_ext.convex_sort(pts, masks, circular)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        return ()


def convex_sort(pts, masks, circular=True):
    return ConvexSortFunction.apply(pts, masks, circular)


