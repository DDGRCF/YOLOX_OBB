import torch
import torch.nn as nn
from .parse_model import get_model
from yolox.utils import check_anchor_order
from .modules import Detect

class Model(nn.Module):
    def __init__(self, modules_cfg="", losses_cfg="", in_channel=3, num_classes=80, max_stride=256):
        super().__init__()
        self.model, self.save = get_model(modules_cfg, in_channel=in_channel, num_classes=num_classes)
        self.in_channel = in_channel
        self.is_export_onnx = False 
        detect_module = self.model[-1]
        detect_module.initialize_losses(losses_cfg)
        self.num_classes = getattr(detect_module, "num_classes", 80)
        if not hasattr(detect_module, "strides"):
            self.set_strides(self.model, in_channel, max_stride=max_stride)
        else:
            with torch.no_grad():
                self.forward(torch.zeros(1, in_channel, max_stride, max_stride))
        if hasattr(detect_module, "anchors"):
            self.adjust_anchors(self.model)


    @torch.no_grad()
    def set_strides(self, model, in_channel, max_stride=256):
        model[-1].strides = torch.tensor([max_stride / x.shape[-2] 
                for x in self.forward(torch.zeros(1, in_channel, max_stride, max_stride))])
        self.strides = model[-1].strides
    
    @torch.no_grad()
    def adjust_anchors(self, model):
        detect = model[-1]
        if not isinstance(detect.anchors, int or float) and check_anchor_order(detect.anchors_grids, detect.strides):
            detect.anchors.flip_(0)
            detect.anchors_grids.flip_(0)
    
    def initialize_weights(self):
        # init for yolo
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
        # detect init
        self.model[-1].initialize_biases()
    
    def forward(self, x):
        return self.forward_once(x)

    def get_losses(self, label, xin):
        output = self.forward(xin)
        return self.model[-1].get_losses(label, output)

    def forward_once(self, x):
        # y, dt = [], []  # outputs
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # if visualize:
            #     feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def postprocess(self, *args, **kwargs):
        return self.model[-1].postprocess(*args, **kwargs)




class CondInstModel(Model):
    def __init__(self, modules_cfg="", losses_cfg="", in_channel=3, num_classes=80):
        super().__init__(modules_cfg, losses_cfg, in_channel, num_classes)

    @torch.no_grad()
    def set_strides(self, model, in_channel, max_stride=256):
        model[-1].strides = torch.tensor([max_stride / x.shape[-2] 
                for x in self.forward(torch.zeros(1, in_channel, max_stride, max_stride))][:-1])
        self.strides = model[-1].strides

    @torch.no_grad()
    def adjust_anchors(self, model):
        pass

    def get_losses(self, label, xin):
        output = self.forward(xin)
        return self.model[-1].get_losses(label, (output, xin))
