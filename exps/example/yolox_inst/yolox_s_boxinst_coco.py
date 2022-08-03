#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import itertools
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import MaskExp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.modules_config = "configs/modules/boxinst_darknet_simplify.yaml"
        self.losses_config = "configs/losses/boxinst_losses.yaml"
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
            mask_thre=0.50,
        )
        self.eval_interval = 5
        self.clip_norm_val = 0.0
        # Onnx export
        self.include_post = True
        self.export_input_names = ["input"]
        self.export_output_names = ["masks", "bboxes"]
        # Debug
        self.enable_debug = False
        # self._get_data_info(self.dataset_config)

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            MosaicBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img)

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=100, # 120 -> 100
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = MosaicBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

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
                              if clip_norm_val > 0.0 else optim)(pg, lr=lr, momentum=0.9, weight_decay=self.weight_decay)

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

    def get_data_prefetcher(self, train_loader, data_type):
        from yolox.data import DataPrefetcher
        return DataPrefetcher(train_loader, data_type)

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def model_wrapper(self, model):
        import torch.nn.functional as F
        from yolox.models import SiLU
        from yolox.utils import replace_module
        model = replace_module(model, nn.SiLU, SiLU)

        def postprocess(output, num_classes=0.45, conf_thre=0.01, mask_thre=0.45):
            bs_masks, bs_bboxes = output[0][0], output[0][0]
            bs_scores = bs_bboxes[:, 4] * bs_bboxes[:, 5]
            bs_labels = bs_bboxes[:, -1]
            bs_bboxes = bs_bboxes[:, :4]
            keep = bs_scores > conf_thre
            bs_scores = bs_scores[keep]
            bs_labels = bs_labels[keep]
            bs_masks = bs_masks[keep]
            bs_bboxes = bs_bboxes[keep]
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
