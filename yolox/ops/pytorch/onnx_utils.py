import torch

def _size_helper(g, self, dim):
    full_shape = g.op('Shape', self)
    from torch.onnx.symbolic_opset9 import select
    return select(g, full_shape, g.op('Constant', value_t=torch.tensor([0])),
                  dim)