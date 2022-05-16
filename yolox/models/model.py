import torch
import torch.nn as nn
from .parse_model import get_model
from yolox.utils import check_anchor_order
from .modules import Detect

class Model(nn.Module):
    def __init__(self, modules_cfg="", losses_cfg="", in_channel=3, num_classes=80):
        super().__init__()
        self.model, self.save = get_model(modules_cfg, in_channel=in_channel, num_classes=num_classes)
        self.in_channel = in_channel
        self.is_export_onnx = False 
        detect_module = self.model[-1]
        detect_module.initialize_losses(losses_cfg)
        self.num_classes = getattr(detect_module, "num_classes", 80)
        if isinstance(detect_module, Detect):
            max_stride = 256
            if not hasattr(detect_module, "strides"):
                with torch.no_grad():
                    detect_module.strides = torch.tensor([max_stride / x.shape[-2] for x in self.forward(torch.zeros(1, in_channel, max_stride, max_stride))])
            else:
                with torch.no_grad():
                    self.forward(torch.zeros(1, in_channel, max_stride, max_stride))
                # detect_module.strides = [max_stride / x.shape[-2] for x in self.forward(torch.zeros(1, self.in_channel, max_stride, max_stride))]
            if hasattr(detect_module, "anchors") and \
                check_anchor_order(detect_module.anchors_grids, detect_module.strides):
                detect_module.anchors.flip_(0)
                detect_module.anchors_grids.flip_(0)
            self.strides = detect_module.strides
    
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



