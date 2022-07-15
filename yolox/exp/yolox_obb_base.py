#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import torch
import torch.distributed as dist
import torch.nn as nn

from .yolox_base import Exp

class OBBExp(Exp):
    def __init__(self):
        super().__init__()
        self.modules_config = "configs/modules/yoloxs_obb.yaml"
        self.losses_config = "configs/losses/yolox_losses_obb.yaml"
        self.dataset_config="configs/datasets/dota10.yaml"
        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (1024, 1024)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.copy_paste_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.4, 1.2)
        self.mixup_scale = (0.4, 1.2)
        self.shear = 2.
        self.enable_debug = False
        self.enable_resample = False
        self.aug_ignore = None
        self.empty_ignore=True
        self.long_wh_thre=6
        self.short_wh_thre=3
        self.overlaps_thre=0.6
        # --------------  training config --------------------- #
        self.warmup_epochs = 1
        self.max_epoch = 80
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 3
        self.min_lr_ratio = 0.05
        self.ema = True
        self.no_eval = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.output_dir = "YOLOX_outputs"
        self.postprocess_cfg = dict(
            conf_thre=0.05,
            nms_thre=0.1,
        )
        self.test_size = (1024, 1024)
        self.evaluate_cfg = dict(
            is_submiss=False,
            is_merge=False,
            nproc=1
        )
        # self._get_data_info(self.dataset_config)

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import (
            DOTADataset,
            OBBTrainTransform,
            MosaicBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicOBBDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = DOTADataset(
                name=self.train_ann,
                data_dir=self.data_dir,
                img_size=self.input_size,
                preproc=OBBTrainTransform(
                    max_labels=400,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicOBBDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=OBBTrainTransform(
                max_labels=350,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                long_wh_thre=self.long_wh_thre,
                short_wh_thre=self.short_wh_thre,
                overlaps_thre=self.overlaps_thre),
            overlaps_thre=self.overlaps_thre,
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_resample=self.enable_resample,
            mosaic_prob=self.mosaic_prob,
            copy_paste_prob=self.copy_paste_prob,
            mixup_prob=self.mixup_prob,
            enable_debug=self.enable_debug,
            aug_ignore = self.aug_ignore,
            empty_ignore=self.empty_ignore,
            work_dir=os.path.join(self.output_dir, self.exp_name)
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

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            if targets.shape[-1] == 6:
                targets[..., 1:-1:2] = targets[..., 1:-1:2] * scale_x
                targets[..., 2:-1:2] = targets[..., 2:-1:2] * scale_y
            else:
                targets[..., 1::2] = targets[..., 1::2] * scale_x
                targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import DOTADataset, ValTransform
        valdataset = DOTADataset(
            data_dir=self.data_dir,
            name=self.val_ann if not testdev else self.test_ann,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            save_results_dir=os.path.join(self.output_dir, self.exp_name, "test_results")
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import DOTAEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = DOTAEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            num_classes=self.num_classes,
            testdev=testdev,
            **self.postprocess_cfg
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(
            model, is_distributed, half, **self.evaluate_cfg)
