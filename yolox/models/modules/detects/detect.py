import torch
import numpy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from yolox.utils import postprocess as postprocess_
from yolox.models.losses import *
class Detect(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def get_losses(*args):
        logger.info("Get Losses Func is Empty")

    def initialize_losses(self, cfg):
        import yaml, re
        if isinstance(cfg, str):
            with open(cfg) as f:
                loss_config = yaml.safe_load(f)
        elif isinstance(cfg, dict):
            loss_config = cfg
        functions = loss_config["Functions"]
        args = loss_config["Args"]
        if functions is not None:
            for _, (name, arg) in enumerate(functions.items()):
                func = eval(arg["func"])
                a = []
                k = {}
                for a_ in arg["args"]:
                    if isinstance(a_, str) and re.match("^(kwargs\()(.*)(\))$", a_) is not None:
                        a_ = "dict" + a_.lstrip("kwargs")
                        k.update(eval(a_)) 
                    else:
                        a.append(a_)
                func_init = func(*a, **k)
                setattr(self, name, func_init)
        if args is not None:
            for _, (name, arg) in enumerate(args.items()):
                setattr(self, name, arg)

    @staticmethod
    def postprocess(*args, **kwargs):
        return postprocess_(*args, *kwargs)
    
    @staticmethod
    def _make_grid(wsize, hsize, out_shape=(), dtype=torch.float32):
        out_shape =  out_shape if len(out_shape) else (1, 1, wsize, hsize, 2)
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        return torch.stack((xv, yv), 2).view(out_shape).type(dtype)