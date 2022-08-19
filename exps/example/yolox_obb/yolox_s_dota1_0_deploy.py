#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from yolox.exp import OBBExp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 300
        self.no_aug_epochs = 15
        self.warmup_epochs = 5
        self.no_eval = True
        self.mosaic_prob = 1.0
        self.copy_paste_prob = 1.0
        self.mixup_prob = 0.0
        self.enable_resample = True # for resampling samples
        # enable debug which allow usr to debug aug images
        self.enable_debug = False
        # ignore images which exists horizontal labels, 
        # the rotated aug will not implement the classes by adding this item
        self.aug_ignore = ['roundabout', 'storage-tank'] 
        # ignore images which has no labels, which ensure each train contains labels
        self.empty_ignore = True
        self.evaluate_cfg = dict(
            is_submiss=False,
            is_merge=False,
            nproc=10)
        self.postprocess_cfg = dict(
            conf_thre=0.05,
            nms_thre=0.1,
        )
        # Model export (onnx name)
        self.export_input_names = ["input"]
        self.export_output_names = ["boxes", "scores", "class"]
        self.include_post = True

    def model_wrapper(self, model):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from yolox.utils import replace_module
        from yolox.models import SiLU
        model = replace_module(model, nn.SiLU, SiLU)

        # only support static
        def postprocess(prediction, num_classes=15, **kwargs):
            boxes = prediction[0, :, :5]
            obj_score = prediction[0, :, 5]
            cls_out = prediction[0, :, 6: 6 + num_classes]
            cls_score, cls_pred = torch.max(cls_out, 1)
            final_score = obj_score * cls_score
            cls_pred = cls_pred.float()
            return boxes, final_score, cls_pred

        class OModel(nn.Module):
            def __init__(self, model, num_classes, postprocess_cfg, include_post=False):
                super().__init__()
                self.main_model = model
                self.include_post = include_post
                self.num_classes = num_classes
                self.postprocess_cfg = postprocess_cfg

            def forward(self, input):
                output = self.main_model(input)
                if self.include_post:
                    output = postprocess(output, self.num_classes, **self.postprocess_cfg)
                return output
        
        return OModel(model, self.num_classes, self.postprocess_cfg, self.include_post)



        
