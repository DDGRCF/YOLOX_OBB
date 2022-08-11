#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict
from tabulate import tabulate

import torch
from torch.nn import Module
from loguru import logger

from yolox.utils import LRScheduler

class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""

    def __init__(self):
        self.seed = None
        self.output_dir = "./YOLOX_outputs"
        self.print_interval = 100
        self.eval_interval = 10
        self.__no_print_buffers = ["class_names"]
    
    def _get_data_info(self, cfg):
        if isinstance(cfg, str):
            import yaml
            with open(cfg, "r") as f:
               cfg_dict = yaml.safe_load(f) 
        elif isinstance(cfg, dict):
            cfg_dict = cfg
        else:   
            raise NotImplementedError
        self.__dict__.update(cfg_dict)
    
    def model_wrapper(self, model):
        return model
    
    @abstractmethod
    def get_data_prefetcher(self, train_loader):
        pass

    @abstractmethod
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_data_loader(
        self, batch_size: int, is_distributed: bool
    ) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_lr_scheduler(
        self, lr: float, iters_per_epoch: int, **kwargs
    ) -> LRScheduler:
        pass

    @abstractmethod
    def get_evaluator(self):
        pass

    @abstractmethod
    def update_LR(self, lr_schedule, optimizer, t_iter, c_iter, epoch):
        pass

    @abstractmethod
    def eval(self, model, evaluator, weights):
        pass

    def merge(self, args):
        if args is not None:
            for arg_name in args:
                arg_value = args[arg_name]
                if hasattr(self, arg_name):
                    assert not isinstance(arg_value, Dict), \
                    "Unsupport Dict Type Set"
                    check_parsers(self, arg_name, arg_value, mode="tuple")
                else:
                    check_parsers(self, arg_name, arg_value, mode="dict")

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if (not k.startswith("_") and k not in self.__no_print_buffers) #TODO: Norm
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def train_in_epoch_deal(self, *args, **kwargs):
        pass

    def train_before_epoch_deal(self, *args, **kwargs):
        pass

    def train_after_epoch_deal(self, *args, **kwargs):
        pass

    def train_before_iter_deal(self, *args, **kwargs):
        pass

    def train_one_iter_deal(self, *args, **kwargs):
        pass

    def train_after_iter_deal(self, *args, **kwargs):
        pass

def _pair(value):
    if isinstance(value, tuple):
        if len(value) == 1:
            value = value * 2
        return value
    elif isinstance(value, list):
        return tuple(value)
    else:
        return (value, value)
    
def check_parsers(exp, name, value, mode="dict"):
    if mode == "dict":
        for exp_name, exp_value in vars(exp).items():
            if isinstance(exp_value, Dict):
                for n in exp_value:
                    if name == n:
                        exp_value[n] = value
                        setattr(exp, exp_name, exp_value)
                        return 0
    elif mode == "tuple":
        if isinstance(getattr(exp, name, None), list) or \
            isinstance(getattr(exp, name, None), tuple):
            setattr(exp, name, _pair(value))
        else:
            setattr(exp, name, value)
    else:
        raise NotImplemented
    return 0