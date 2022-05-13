#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from .base_exp import BaseExp, load_info_wrapper
from .yolox_base import Exp
from .yolox_obb_base import OBBExp
from .build import get_exp