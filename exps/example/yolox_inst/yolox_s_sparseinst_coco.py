#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import MaskExp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_num_workers = 0
        self.max_epoch = 80
        self.no_aug_epochs = 5
        self.no_eval = True
        self.mosaic_prob = 1.0
        self.copy_paste_prob = 1.0
        # enable debug which allow usr to debug aug images
        self.enable_debug = True
        # ignore images which exists horizontal labels, 
        # the rotated aug will not implement the classes by adding this item
        # ignore images which has no labels, which ensure each train contains labels
        self.empty_ignore = True
        self.test_conf = 0.05
        self._get_data_info(self.dataset_config)
