#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

# from yolox.exp import OBBExp as MyExp
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = "yolov5s"
        self.modules_config = "configs/modules/yolov5s.yaml"
        self.losses_config = "configs/losses/yolov5_losses.yaml"
        self.data_num_workers = 4

        self.max_epoch = 100
        self.no_aug_epochs = 5
        self.no_eval = False
        self.enable_mixup = True
        self.enable_copy_paste = True
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        # enable debug which allow usr to debug aug images
        self.enable_debug = False
