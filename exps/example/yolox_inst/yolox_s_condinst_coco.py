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
        self.modules_config = "configs/modules/condinst_darknet_simplify.yaml"
        self.losses_config = "configs/losses/condinst_losses.yaml"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.max_epoch = 36
        self.no_aug_epochs = 2
        self.warmup_epochs = 1
        self.basic_lr_per_img = 0.01 / 64.0
        self.min_lr_ratio = 0.05
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.scheduler = "yoloxwarmcos"

        self.data_num_workers = 4
        self.no_eval = False
        # mosaic data augmentation | default set 0.0 for fast train
        self.mosaic_prob = 0.0
        # copy paste data augmentation | default set 0.0 for fast train
        self.copy_paste_prob = 0.0
        self.postprocess_cfg = dict(
            conf_thre=0.05,
            mask_thre=0.45,
        )
        self.include_post = True
        self.eval_interval = 5
        self.clip_norm_val = 0.0
        self.export_input_names = ["input"]
        self.export_output_names = ["masks", "bboxes"]
        # Debug
        self.enable_debug = False
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
                    v.eval()
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg.append(v.weight)  # apply decay

            optim = torch.optim.SGD
            clip_norm_val = self.clip_norm_val
            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            self.optimizer = (FullModelGradientClippingOptimizer 
                              if clip_norm_val > 0.0 else optim)(pg, lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)

        return self.optimizer

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            num_classes=self.num_classes,
            testdev=testdev,
            with_bbox=True,
            with_mask=True,
            metric = ["bbox","segm"],
            save_metric="segm",
            **self.postprocess_cfg
        )
        return evaluator

    def get_model(self):
        from yolox.models import CondInstModel as Model
        self.model = Model(self.modules_config, self.losses_config, in_channel=3, num_classes=self.num_classes)
        self.model.initialize_weights()
        assert len(self.class_names) == self.num_classes
        assert self.model.num_classes == self.num_classes
        return self.model

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def model_wrapper(self, model):
        from yolox.models import SiLU
        from yolox.utils import replace_module
        model = replace_module(model, nn.SiLU, SiLU)

        def postprocess(output, num_classes=80, conf_thre=0.01, mask_thre=0.45, **kwargs):
            bs_masks, bs_bboxes = output[0][0], output[1][0]
            bs_scores = bs_bboxes[:, 4] * bs_bboxes[:, 5]
            bs_labels = bs_bboxes[:, -1]
            bs_bboxes = bs_bboxes[:, :4]
            bs_masks = (bs_masks > mask_thre).type(bs_scores.dtype)
            return bs_masks, torch.cat((bs_bboxes, bs_scores[:, None], bs_labels[:, None]), dim=1)
        
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



