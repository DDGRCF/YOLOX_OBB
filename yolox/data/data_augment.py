#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np
import shapely.geometry as shgeo
import BboxToolkit as bt

from yolox.utils import xyxy2cxcywh
# from BboxToolkit import poly2hbb


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )

def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale

def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets

def apply_affine_to_obboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    targets_ = np.ones((4 * num_gts, 3))
    targets_[:, :2] = targets[:, :8].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    targets_ = targets_ @ M.T  # apply affine transform
    targets_ = targets_.reshape(num_gts, 8)
    targets[:, :8] = targets_

    return targets

def box_candidates(labels, high_limit_size, low_limit_size=(-1, -1), return_inds=False):
    assert labels.shape[-1] >= 8
    ctr_x = np.mean(labels[..., 0:8:2], axis=-1)
    ctr_y = np.mean(labels[..., 1:8:2], axis=-1)
    mask_x = np.logical_and(low_limit_size[1] < ctr_x, ctr_x < high_limit_size[1])
    mask_y = np.logical_and(low_limit_size[0] < ctr_y, ctr_y < high_limit_size[0])
    mask_labels = np.logical_and(mask_x, mask_y)
    mask_inds = np.nonzero(mask_labels)[0]
    labels = labels[mask_inds]
    if return_inds:
        return labels, mask_inds
    else:
        return labels

def obox_candidates(obboxes, high_limit_size, low_limit_size=(-1, -1), overlaps_thre=0.6, return_inds=False):
    image_obbox = np.asarray(
        [[low_limit_size[1], low_limit_size[0], high_limit_size[1], high_limit_size[0]]], dtype=obboxes.dtype)
    
    overlaps = bt.bbox_overlaps(obboxes, image_obbox, mode="iof", is_aligned=False).squeeze(-1)
    mask = overlaps > overlaps_thre
    mask_inds = np.nonzero(mask)[0]
    obboxes = obboxes[mask_inds]
    if return_inds:
        return obboxes, mask_inds
    else:
        return obboxes

def mask_random_affine(
    img,
    targets=(),
    masks=None,
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=1.0,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets):
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)
        masks = cv2.warpAffine(masks, M, dsize=target_size, borderValue=0)
        if masks.ndim == 2:
            masks = masks[..., None]

    return img, targets, masks

def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets

def obb_random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))
    if len(targets) > 0:
        targets = apply_affine_to_obboxes(targets, target_size, M, scale)

    return img, targets

def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        if boxes.shape[-1] == 4: # horizontal box
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        elif boxes.shape[-1] == 5:
            boxes[:, 0] = width - boxes[:, 0]
            boxes[:, 4] *= -1
        elif boxes.shape[-1] == 8: # rotated box
            boxes[:, 0::2] = width - boxes[:, 6::-2]

    return image, boxes

def _mask_mirror(image, boxes, prob=0.5, masks=None):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        if masks is not None:
            masks = masks[:, ::-1]
    return image, boxes, masks


def preproc(img, input_size, swap=(2, 0, 1), padding_value=114):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], img.shape[-1]), dtype=np.uint8) * padding_value
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * padding_value

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    origin_dim = img.ndim
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    resized_dim = resized_img.ndim
    if origin_dim != resized_dim:
        resized_img = resized_img[..., None]
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    if swap is not None:
        padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels



class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))


class OBBTrainTransform:
    def __init__(self, max_labels=100, flip_prob=0.5, 
                 hsv_prob=1.0, long_wh_thre=8, 
                 short_wh_thre=3, overlaps_thre=0.6):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.long_wh_thre=long_wh_thre
        self.short_wh_thre=short_wh_thre
        self.overlaps_thre=overlaps_thre 

    def __call__(self, image, targets, input_dim):
        last_axis = targets.shape[-1]
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, last_axis), dtype=np.float32)
            image, _ = preproc(image, input_dim)
            return image, targets

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        # height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        boxes[:, :4] = boxes[:, :4] * r_
        # select box which's sides > 1 pixel
        _, height, width = image_t.shape
        boxes_image_overlaps = bt.bbox_overlaps(boxes[:, :5], 
                np.asarray([[width / 2, height / 2, width, height, 0]], dtype=boxes.dtype), 
                mode="iof", is_aligned=False)
        mask_b = boxes_image_overlaps.squeeze(-1) > self.overlaps_thre
        mask_short_wh = np.minimum(boxes[:, 2], boxes[:, 3]) > self.short_wh_thre
        mask_long_wh = np.maximum(boxes[:, 2], boxes[:, 3]) > self.long_wh_thre
        mask_b = np.logical_and.reduce(np.array((mask_b, mask_short_wh, mask_long_wh)))
            
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            targets = np.zeros((self.max_labels, last_axis), dtype=np.float32)
            return image_t, targets

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, last_axis))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size, *args, **kwargs):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 6))

class MaskTrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, wh_thre=8):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.wh_thre = wh_thre

    def _mask_filter(self, masks, keep_inds=None, is_tolist=False):
        if keep_inds is not None:
            masks = masks[..., keep_inds]
        if is_tolist:
            return [masks[..., i] for i in range(masks.shape[-1])]
        else:
            return masks

    def __call__(self, image, bbox_targets, input_dim, mask_targets=None):
        num_targets = len(bbox_targets)
        boxes = bbox_targets[:, :4].copy()
        labels = bbox_targets[:, 4].copy()
            
        if len(boxes) == 0:
            bbox_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image = preproc(image, input_dim)[0]
            mask_targets = np.zeros((0, input_dim[0], input_dim[1]), dtype=np.float32)
            return image, bbox_targets, mask_targets

        image_o = image.copy()
        bbox_targets_o = bbox_targets.copy()
        masks_o = mask_targets.copy()
        boxes_o = bbox_targets_o[:, :4]
        labels_o = bbox_targets_o[:, 4]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)

        image_t, boxes, masks = _mask_mirror(image, boxes, self.flip_prob, masks=mask_targets)
        image_t, r_ = preproc(image_t, input_dim)

        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        filter_b = np.minimum(boxes[:, 2], boxes[:, 3]) > self.wh_thre
        filter_m = masks.sum((0, 1)) > self.wh_thre ** 2
        filter_b = np.bitwise_and(filter_b, filter_m)
        boxes_t = boxes[filter_b]
        labels_t = labels[filter_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            masks_t = preproc(masks_o, (input_dim[0], input_dim[1], num_targets), None, 0.)[0]
            masks_t = self._mask_filter(masks_t, None, is_tolist=False)
        else:
            keep_inds = np.nonzero(filter_b)[0]
            masks = preproc(masks, (input_dim[0], input_dim[1], num_targets), None, 0.)[0]
            masks_t = self._mask_filter(masks, keep_inds, is_tolist=False)

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, targets_t.shape[-1]))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        if isinstance(masks_t, list):
            masks_t = masks_t[ :self.max_labels]
        elif isinstance(masks_t, np.ndarray):
            masks_t = masks_t[..., :self.max_labels]

        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        masks_t = (masks_t > 0).astype(np.float32)

        return (image_t, 
                padded_labels, 
                masks_t.transpose(2, 0, 1))