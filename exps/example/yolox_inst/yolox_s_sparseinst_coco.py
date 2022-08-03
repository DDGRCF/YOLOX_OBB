#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import itertools
import torch
import torch.nn as nn


from yolox.exp import MaskExp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.modules_config = "configs/modules/sparseinst_darknet_simplify.yaml"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 24
        self.no_aug_epochs = 2
        self.data_num_workers = 4
        self.no_eval = False
        # mosaic data augmentation | default set 0.0 for fast train
        self.mosaic_prob = 0.0
        # copy paste data augmentation | default set 0.0 for fast train
        self.copy_paste_prob = 1.0
        self.basic_lr_per_img = 5.0e-5 / 64.0
        self.weight_decay = 0.05
        self.postprocess_cfg = dict(
            conf_thre=0.005,
            mask_thre=0.45,
        )
        # LR Scheduler
        self.scheduler = "multistep"
        self.milestones = (16, 20)
        self.clip_norm_val = 0.0
        self.eval_interval = 2
        # Debug
        self.enable_debug = False
        # Model export (onnx name)
        self.export_input_names = ["input"]
        self.export_output_names = ["masks", "scores"]
        self.include_post = True
        # self._get_data_info(self.dataset_config)

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            lr = self.basic_lr_per_img * batch_size

            pg = []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if isinstance(v, nn.Identity):
                    continue
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg.append(v.weight);
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg.append(v.weight)  # apply decay

            optim = torch.optim.AdamW
            clip_norm_val = self.clip_norm_val
            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            # NOTE:if add fp16 to model, adam optimizer should be added (eps=1e-3), otherwise it is easy to appear nan
            self.optimizer = (FullModelGradientClippingOptimizer 
                              if clip_norm_val > 0.0 else optim)(pg, lr=lr, amsgrad=False, weight_decay=self.weight_decay)

        return self.optimizer

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            num_classes=self.num_classes,
            testdev=testdev,
            with_bbox=False,
            with_mask=True,
            metric = ["segm"],
            save_metric="segm",
            **self.postprocess_cfg
        )
        return evaluator

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            milestones=self.milestones,
        )
        return scheduler
    
    def model_wrapper(self, model):
        import torch.nn.functional as F
        from yolox.utils import replace_module
        from yolox.models import SiLU, AdaptiveAvgPool2d__forward

        def replace_func(rep_module, new_module, **kwargs):
            rep_module.forward = AdaptiveAvgPool2d__forward
            return rep_module

        model = replace_module(model, nn.SiLU, SiLU)
        model = replace_module(model, nn.AdaptiveAvgPool2d, None, replace_func=replace_func)

        # for 1 batchsize
        def postprocess(output, num_classes=80, 
                        conf_thre=0.01, mask_thre=0.45, 
                        scale_factor=4, eps=1e-6, **kwargs):
            bs_masks, bs_scores = output[0][0], output[1][0]
            bs_scores, bs_labels = bs_scores.max(dim=-1)
            keep = bs_scores > conf_thre
            bs_scores = bs_scores[keep]
            bs_labels = bs_labels[keep]
            bs_masks = bs_masks[keep]
            bs_masks_ = (bs_masks > mask_thre).float()
            bs_scores = bs_scores * ((bs_masks_ * bs_masks).sum((1, 2)) / (bs_masks_.sum((1, 2)) + eps))
            bs_masks = F.interpolate(bs_masks[:, None], scale_factor=scale_factor, 
                                mode="bilinear", align_corners=False).squeeze(1)
            bs_masks = (bs_masks > mask_thre).type(bs_scores.dtype)
            bs_scores = torch.cat((bs_scores, bs_labels.type(bs_scores.dtype)), dim=-1)
            return bs_masks, bs_scores

        # for OModel
        class OModel(nn.Module):
            """
                model
            """
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
