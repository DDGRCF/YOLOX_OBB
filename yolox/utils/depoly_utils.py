import torch
import torch.nn as nn
from yolox.utils import postprocess, obbpostprocess

class DepolyModel(nn.Module):
    def __init__(self, model, exp, **kwargs):
        super().__init__()
        self.model = model
        self.postprocess_func = getattr(model, "postprocess", postprocess)
        self.postprocess_cfg = exp.postprocess_cfg
        self.num_classes = exp.num_classes
        self.kwargs = kwargs

    def forward(self, x):
        x = self.model(x)
        x = self.postprocess_func(x, self.num_classes, **self.postprocess_cfg, **self.kwargs)[0]
        return x

        

