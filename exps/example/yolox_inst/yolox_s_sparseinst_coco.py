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
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 12
        self.no_aug_epochs = 2
        self.data_num_workers = 4
        self.no_eval = False
        # mosaic data augmentation | default set 0.0 for fast train
        self.mosaic_prob = 0.0
        # copy paste data augmentation | default set 0.0 for fast train
        self.copy_paste_prob = 0.0
        self.basic_lr_per_img = 4.0e-5 / 64.0
        self.weight_decay = 0.05
        self.postprocess_cfg = dict(
            conf_thre=0.01,
            mask_thre=0.45,
        )
        # LR Scheduler
        self.scheduler = "multistep"
        self.milestones_epoch_step = (7, 10)
        self.clip_norm_val = 0.0
        self.eval_interval = 3
        # Debug
        self.enable_debug = False
        self._get_data_info(self.dataset_config)

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

            optim = torch.optim.AdamW
            clip_norm_val = self.clip_norm_val
            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

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
        if not hasattr(self, "milestones"):
            assert hasattr(self, "milestones_epoch_step")
            self.milestones = []
            for step_epoch in self.milestones_epoch_step:
                self.milestones.append(step_epoch * iters_per_epoch)

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            milestones=self.milestones,
        )
        return scheduler
