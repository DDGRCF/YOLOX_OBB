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

from yolox.utils import xyxy2cxcywh
from BboxToolkit import poly2hbb


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

def box_ioa_filter(t_labels, o_labels, ioa_thre=0.30, eps=1e-6):
    assert o_labels.ndim >= 2
    if t_labels.ndim == 1:
        t_labels = t_labels[None]
    poly_t = t_labels[..., :8]
    poly_o = o_labels[..., :8]
    hbb_t = poly2hbb(poly_t).reshape(-1, 1, 2)
    hbb_o = poly2hbb(poly_o).reshape(1, -1, 2)
    lt = np.maximum(hbb_t[..., :2], hbb_o[..., :2])
    rb = np.minimum(hbb_t[..., 2:], hbb_o[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf) # (#num_t, #num_o, 2)
    overlaps = wh[..., 0] * wh[..., 1] # (#num_t, #num_o)
    areas_t = (hbb_t[..., 2] - hbb_t[..., 0]) * (hbb_t[..., 3] - hbb_t[..., 1]) # (#num_t, 1)
    keep_inds = np.nonzero(((overlaps / (areas_t + eps)) < ioa_thre).sum(1) == 0.0)[0]
    return t_labels[keep_inds]


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

def obb_random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, :8].reshape(n * 4, 2) # for r bbox
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)
        # i = box_candidates(box1=targets[:, :8].T * s, box2=xy.T)
        # targets = targets[i] 
        # targets[:, :8] = xy[i]
        targets[:, :8] = xy

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


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

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
    def __init__(self, max_labels=100, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        last_axis = targets.shape[-1]
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, last_axis), dtype=np.float32)
            image, _ = preproc(image, input_dim)
            return image, targets

        # image_o = image.copy()
        # targets_o = targets.copy()
        # # height_o, width_o, _ = image_o.shape
        # boxes_o = targets_o[:, :-1]
        # labels_o = targets_o[:, -1]

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        # height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        boxes[:, :4] = boxes[:, :4] * r_
        # select box which's sides > 1 pixel
        if last_axis == 5 or last_axis == 6:
            _, height, width = image_t.shape
            mask_b_wh = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
            mask_b_x = np.logical_and(0 < boxes[:, 0], boxes[:, 0] < width) 
            mask_b_y = np.logical_and(0 < boxes[:, 1], boxes[:, 1] < height) 
            mask_b = np.logical_and.reduce(np.array((mask_b_wh, mask_b_x, mask_b_y)))
        elif last_axis == 9:
            _, height, width = image_t.shape
            side_l1 = np.sqrt((boxes[:, 0] - boxes[:, 2]) ** 2 + (boxes[:, 1] - boxes[:, 3]) ** 2)
            side_l2 = np.sqrt((boxes[:, 2] - boxes[:, 4]) ** 2 + (boxes[:, 3] - boxes[:, 5]) ** 2)
            b_ctrx, b_ctry = np.mean(boxes[:, 0:8:2], axis=1), np.mean(boxes[:, 1:8:2], axis=1)
            mask_b_x = np.logical_and(0 < b_ctrx, b_ctrx < width)
            mask_b_y = np.logical_and(0 < b_ctry, b_ctry < height)
            mask_b_wh = np.minimum(side_l1, side_l2) > 1
            mask_b = np.logical_and.reduce(np.array((mask_b_x, mask_b_y, mask_b_wh)))
            
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        # if len(boxes_t) == 0:
        #     image_t, r_o = preproc(image_o, input_dim)
        #     boxes_o[:, :4] = boxes_o[:, :4] * r_o
        #     boxes_t = boxes_o
        #     labels_t = labels_o
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
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 6))