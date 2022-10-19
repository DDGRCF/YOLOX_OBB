#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from yolox.exp import OBBExp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 80
        self.no_aug_epochs = 2
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

        # deploy
        self.export_input_names = ["input"]
        self.export_output_names = ["boxes", "scores", "class"]
        self.include_post = True
    
        # if use dynamic axes, please use under codes

        # num_proposal = 0
        # for stride in [8, 16, 32]:
        #     num_proposal += int(self.test_size[0] / stride) * int(self.test_size[1] / stride)

        # self.export_trt_dynamic_axes = {
        #     "input": {
        #         "min": [1, 3, *self.test_size],
        #         "media": [1, 3, *self.test_size],
        #         "max": [8, 3, *self.test_size],
        #     },
        #     "boxes": {
        #         "min": [1, num_proposal, 5],
        #         "media": [1, num_proposal, 5],
        #         "max": [8, num_proposal, 5]
        #     },
        #     "scores": {
        #         "min": [1, num_proposal], 
        #         "media": [1, num_proposal],
        #         "max": [8, num_proposal]
        #     },
        #     "class": {
        #         "min": [1, num_proposal], 
        #         "media": [1, num_proposal],
        #         "max": [8, num_proposal]
        #     }
        # }

        # self.export_dynamic_axes = {"input": {0: "batch"}, 
        #                             "boxes": {0: "batch"}, 
        #                             "scores": {0: "batch"}, 
        #                             "class": {0: "batch"}}

    def model_wrapper(self, model, backends="tensorrt"):
        import torch
        import torch.nn as nn
        from yolox.utils import replace_module
        from yolox.models import SiLU, Upsample__forward
        
        class TRTModel(nn.Module):
            def __init__(self, model, num_classes, postprocess_cfg, include_post=False):
                super().__init__()
                # def replace_func(rep_module, new_module, **kwargs):
                    # rep_module.forward = Upsample__forward
                # nn.Upsample.forward = Upsample__forward
                model = replace_module(model, nn.SiLU, SiLU)
                # model = replace_module(model, nn.Upsample, None, replace_func)
                self.main_model = model
                self.include_post = include_post
                self.num_classes = num_classes
                self.postprocess_cfg = postprocess_cfg

            def postprocess(self, prediction, num_classes=15, **kwargs):
                boxes = prediction[:, :, :5]
                obj_score = prediction[:, :, 5]
                cls_out = prediction[:, :, 6: 6 + num_classes]
                cls_score, cls_pred = torch.max(cls_out, 2)
                final_score = obj_score * cls_score
                cls_pred = cls_pred.float()
                return boxes, final_score, cls_pred

            def forward(self, input):
                output = self.main_model(input)
                if self.include_post:
                    output = self.postprocess(output, self.num_classes, **self.postprocess_cfg)
                return output
        

        class NCNNModel(nn.Module):
            def __init__(self, model, num_classes, postprocess_cfg, include_post=False):
                super().__init__()

                from yolox.models.modules.detects.obbdetectx import OBBDetectX
                def decode_outputs(ctx, outputs, dtype):
                    grids = []
                    strides = []
                    for (hsize, wsize), stride in zip(ctx.hw, ctx.strides):
                        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
                        grid = torch.stack((xv, yv), 2).view(1, -1, 2) # shape(1, w * h, 2)
                        grids.append(grid)
                        shape = grid.shape[:2]
                        strides.append(torch.full((*shape, 1), stride)) # shape(1, w * h, 1)

                    grids = torch.cat(grids, dim=1).type(dtype) # shape(1, sigma(w*h), 2)
                    strides = torch.cat(strides, dim=1).type(dtype) # same as above
                    strides = strides.repeat(1, 1, 2) # remove influence of unsupported expand in ncnn

                    outputs_xy = (outputs[..., :2] + grids) * strides
                    outputs_wh = torch.exp(outputs[..., 2:4]) * strides
                    outputs_other = outputs[..., 4:]

                    return torch.cat((outputs_xy, outputs_wh, outputs_other), dim=2)

                OBBDetectX.decode_outputs = decode_outputs
                model = replace_module(model, nn.SiLU, SiLU)

                self.main_model = model
                self.include_post = include_post
                self.num_classes = num_classes
                self.postprocess_cfg = postprocess_cfg

            # postprocess for static 
            def postprocess(self, prediction, num_classes=15, **kwargs):
                boxes = prediction[:, :, :5]
                obj_score = prediction[:, :, 5]
                cls_out = prediction[:, :, 6: 6 + num_classes]
                # cls_score, cls_pred = torch.max(cls_out, 2)
                # final_score = obj_score * cls_score
                # cls_pred = cls_pred.float()
                return boxes, obj_score, cls_out
            
            # only support static
            def forward(self, input):
                output = self.main_model(input)
                if self.include_post:
                    output = self.postprocess(output, num_classes=self.num_classes, **self.postprocess_cfg)
                return output
        
        backends_map = {"tensorrt": TRTModel, "onnx": TRTModel, "torchscript": TRTModel, "ncnn": NCNNModel} 
        assert backends in backends_map, f"Unsupport {backends} backends"

        return backends_map[backends](model, self.num_classes, self.postprocess_cfg, self.include_post)