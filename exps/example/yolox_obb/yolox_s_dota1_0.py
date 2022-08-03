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
