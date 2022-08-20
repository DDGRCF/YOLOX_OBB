#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import torch
from .obb_utils import bbox2type
from yolox.ops.pytorch.nms_rotated import multiclass_obb_nms
from yolox.ops.pytorch.nms import multiclass_nms

__all__ = [
    "filter_box",
    "postprocess",
    "obbpostprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy"
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False, **kwargs):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        # conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        bboxes = image_pred[:, :4]
        obj_conf = image_pred[:, [4]]
        cls_out = image_pred[:, 5: 5 + num_classes]
        class_conf, class_pred = torch.max(cls_out, 1, keepdims=True)
        detections = torch.cat((bboxes, obj_conf, class_conf, class_pred.float()), 1)
        # detections = detections[conf_mask]
        mask_nms = multiclass_nms(bboxes, 
                                  class_conf.squeeze(-1), 
                                  class_pred.squeeze(-1), 
                                  score_factors=obj_conf.squeeze(-1), 
                                  iou_thr=nms_thre,
                                  score_thr=conf_thre,
                                  class_agnostic=class_agnostic,
                                  **kwargs)
        detections = detections[mask_nms]
        if not detections.size(0):
            continue

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def obbpostprocess(prediction, num_classes, conf_thre=0.1, nms_thre=0.50, class_agnostic=False, **kwargs):
    outputs = [None for _ in range(len(prediction))]
    for i, pred in enumerate(prediction):
        if not pred.size(0) > 0: 
           continue
        rboxes = pred[:, :5]
        obj_conf = pred[:, [5]]
        cls_out = pred[:, 6: 6 + num_classes]
        class_conf, class_pred = torch.max(cls_out, 1, keepdims=True) 
        rboxes_poly = bbox2type(rboxes, 'poly')
        new_pred = torch.cat([rboxes_poly, obj_conf, class_conf, class_pred], dim=-1)               
        mask_nms = multiclass_obb_nms(rboxes, 
                                    class_conf, 
                                    class_pred, 
                                    score_factors=obj_conf, 
                                    iou_thr=nms_thre, 
                                    score_thr=conf_thre, 
                                    class_agnostic=class_agnostic,
                                    **kwargs)
        new_pred = new_pred[mask_nms]

        if not new_pred.size(0):        
            continue

        if outputs[i] is None:
            outputs[i] = new_pred
        else:
            outputs[i] = torch.cat([outputs[i], new_pred])

    return outputs

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2]) # shape(num_gts, num_pos, 2)
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1) # shape(num_gts)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1) # shape(num_pos)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2) # shape(num_gts, num_pos)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all()) # shape(num_gts, num_pos)
    return area_i / (area_a[:, None] + area_b - area_i) # shape(num_gts, num_pos)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    if bbox.shape[-1] == 4:
        bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    elif bbox.shape[-1] == 8:
        # bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
        # bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
        bbox[:, 0::2] = bbox[:, 0::2] * scale_ratio + padw
        bbox[:, 1::2] = bbox[:, 1::2] * scale_ratio + padh
    return bbox

def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

def cxcywh2xyxy(bboxes):
    x1y1 = bboxes[..., 0:2] - bboxes[..., 2:4] / 2
    x2y2 = bboxes[..., 0:2] + bboxes[..., 2:4] / 2
    return torch.cat((x1y1, x2y2), dim=-1)