import torch
from loguru import logger

def check_anchor_order(anchors, strides):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    assert anchors is not None and strides is not None
    anchors_area = anchors.prod(-1).view(-1)
    delta_anchor_area = anchors_area[-1] - anchors_area[0]
    delta_anchor_stride = strides[-1] - strides[0]
    return 1 if delta_anchor_area.sign() != delta_anchor_stride.sign() else 0