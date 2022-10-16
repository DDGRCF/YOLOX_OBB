#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import torch.nn as nn

from yolox.exp import OBBExp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.dataset_config="configs/datasets/dota20.yaml"
        self.losses_config = "configs/losses/yolox_losses_kld.yaml"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 80
        self.no_eval = True
        self.copy_paste_prob = 1.0
        self.mosaic_prob = 1.0
        self.mixup_prob = 0.0
        # enable debug which allow usr to debug aug images
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

        # deploy
        self.export_input_names = ["input"]
        self.export_output_names = ["boxes", "scores", "class"]
        self.include_post = True

    def model_wrapper(self, model, backends="tensorrt"):
        import torch
        import torch.nn as nn
        from yolox.utils import replace_module
        from yolox.models import SiLU
        
        
        class TRTModel(nn.Module):
            def __init__(self, model, num_classes, postprocess_cfg, include_post=False):
                super().__init__()
            
                model = replace_module(model, nn.SiLU, SiLU)
                self.main_model = model
                self.include_post = include_post
                self.num_classes = num_classes
                self.postprocess_cfg = postprocess_cfg

            # postprocess for static 
            def postprocess(prediction, num_classes=15, **kwargs):
                boxes = prediction[0, :, :5]
                obj_score = prediction[0, :, 5]
                cls_out = prediction[0, :, 6: 6 + num_classes]
                cls_score, cls_pred = torch.max(cls_out, 1)
                final_score = obj_score * cls_score
                cls_pred = cls_pred.float()
                return boxes, final_score, cls_pred

            # only support static
            def forward(self, input):
                output = self.main_model(input)
                if self.include_post:
                    output = self.postprocess(output, self.num_classes, **self.postprocess_cfg)
                return output

        backends_map = {"tensorrt": TRTModel, "onnx": TRTModel, "torchscript": TRTModel} 
        assert backends in backends_map, f"Unsupport {backends} backends"
        
        return backends_map[backends](model, self.num_classes, self.postprocess_cfg, self.include_post)