#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# 禁用远程端口 # TODO: delete
import matplotlib
matplotlib.use("Agg")

import random

import cv2
import numpy as np

import BboxToolkit as bt
from yolox.utils import (adjust_box_anns, get_local_rank, 
                         mintheta_obb, obb2poly, poly2obb_np, bbox2type)

from ..data_augment import (box_candidates, random_affine, 
                                obb_random_perspective)
from .datasets_wrapper import Dataset

def debug_data(labels, img_o, save_name=None, minAreaRect_test=False, dir_name="DEBUG_IMAGES_VIS", save_dir=None):
    import os
    img = img_o.copy()
    if len(labels) == 0:
        return False
    if labels.shape[-1] in [5, 6]:
        import torch
        bboxes = torch.from_numpy(labels[:, 1:])
        # bboxes = obb2poly(bboxes)
        bboxes = bbox2type(bboxes, "poly")
        bboxes = bboxes.numpy()
        if labels.shape[-1] == 6:
            cls = labels[:, 0]
    elif labels.shape[-1] in [8, 9]:
        bboxes = labels[:, :8]
        if labels.shape[-1] == 9:
            cls = labels[:, 8]
    if img.shape[-1] not in [1, 3]:
        img = img.transpose((1, 2, 0))
        img = np.ascontiguousarray(img, dtype=np.int32)
    if save_name is None:
        save_name = '0'
    if save_dir is None:
        save_dir = "YOLOX_outputs"
    save_name_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_name_dir):
        os.makedirs(save_name_dir)
    for bbox, c in zip(bboxes, cls):
        ctr_x = int(np.mean(bbox[0::2]))
        ctr_y = int(np.mean(bbox[1::2]))
        bbox = bbox.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [bbox], True, (0, 255, 255))
        # text = '{}'.format(DOTA_20_CLASSES[int(c)])
        text = '{}'.format(int(c))
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.4, 2)[0]
        cv2.putText(img, text, (ctr_x, ctr_y + txt_size[1]), font, 0.4, (100, 0, 100), thickness=1)
    save_path = os.path.join(save_name_dir, save_name + '.jpg')
    cv2.imwrite(save_path, img)
    return True

def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, perspective=0.0,
        enable_mixup=True, mosaic_prob=1.0, mixup_prob=1.0, 
        *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = get_local_rank()

    def __len__(self):
        return len(self._dataset)

    @Dataset.wrapper_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
            )

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])

            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            return mix_img, padded_labels, img_info, img_id

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels

class MosaicOBBDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, *args, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, perspective=0.0,
        enable_mixup=False, enable_copy_paste=False, 
        enable_resample=False, mosaic_prob=1.0, mixup_prob=1.0, 
        copy_paste_prob=1.0, enable_debug=False, aug_ignore=None, 
        empty_ignore=True, **kwargs
    ):
        """[summary]

        Args:
            dataset ([type]): [description]
            img_size ([type]): [description]
            mosaic (bool, optional): [description]. Defaults to True.
            preproc ([type], optional): [description]. Defaults to None.
            degrees (float, optional): [description]. Defaults to 10.0.
            translate (float, optional): [description]. Defaults to 0.1.
            mosaic_scale (tuple, optional): [description]. Defaults to (0.5, 1.5).
            mixup_scale (tuple, optional): [description]. Defaults to (0.5, 1.5).
            shear (float, optional): [description]. Defaults to 2.0.
            perspective (float, optional): [description]. Defaults to 0.0.
            enable_mixup (bool, optional): [description]. Defaults to False.
            enable_copy_paste (bool, optional): [description]. Defaults to False.
            enable_resample (bool, optional): [description]. Defaults to False.
            mosaic_prob (float, optional): [description]. Defaults to 1.0.
            mixup_prob (float, optional): [description]. Defaults to 1.0.
            copy_paste_prob (float, optional): [description]. Defaults to 1.0.
            enable_debug (bool, optional): [description]. Defaults to False.
            aug_ignore ([type], optional): [description]. Defaults to None.
            empty_ignore (bool, optional): [description]. Defaults to True.
        """

        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.enable_copy_paste = enable_copy_paste
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob
        self.enable_debug = enable_debug
        self.empty_ignore = empty_ignore
        self.enable_resample = enable_resample
        self.empty_ignore = empty_ignore
        self.aug_ignore_list = self._aug_ignore_convert(aug_ignore)
        self.local_rank = get_local_rank()

        self.__dict__.update(kwargs)

    def __len__(self):
        return len(self._dataset)

    def _aug_ignore_convert(self, aug_ignore):
        if aug_ignore is not None:
            assert isinstance(aug_ignore, list) or isinstance(aug_ignore, tuple) 
            if isinstance(aug_ignore[0], str):
                ignore_list = []
                for item in aug_ignore:
                    ignore_list.append(self._dataset.CLASSES.index(item))
            else:
                ignore_list = aug_ignore
            return np.asarray(ignore_list, dtype=np.float32)
        else:
            return None
    
    @Dataset.wrapper_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            AUG_IGNORE_FLAG=False
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]
            # yc, xc = s, s  # mosaic center x, y
            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]
            mosaic_img, mosaic_labels, img_id = self._generate_mosaic_image(input_dim, indices)
            while len(mosaic_labels) == 0 and self.empty_ignore:
                mosaic_img, mosaic_labels, img_id = self._generate_mosaic_image(input_dim)
            if self.aug_ignore_list is not None:
                AUG_IGNORE_FLAG = len(np.intersect1d(self.aug_ignore_list, mosaic_labels[..., -1])) > 0.

            mosaic_img, mosaic_labels = obb_random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees if not AUG_IGNORE_FLAG else 0.,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear if not AUG_IGNORE_FLAG else 0.,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            ) 
            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (
                self.enable_copy_paste
                and not len(mosaic_labels) == 0
                and random.random() < self.copy_paste_prob
            ):
                mosaic_img, mosaic_labels = self.extra_augmention(
                    mosaic_img, mosaic_labels, self.input_dim, "copy_paste",
                    resample=self.enable_resample, choice_prob=1.0)

            if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.extra_augmention(
                    mosaic_img, mosaic_labels, self.input_dim, "mixup")

            loaded_labels = [poly2obb_np(mosaic_label) for mosaic_label in mosaic_labels]
            
            if len(loaded_labels) == 0:
                mosaic_labels = np.zeros((0, 6), dtype=np.float32)
            else:
                loaded_labels = np.concatenate(loaded_labels, axis=0)
                mosaic_labels = mintheta_obb(loaded_labels)

            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])
            if self.enable_debug:
                debug_data(padded_labels, mix_img, str(idx.item()), 
                dir_name="DEBUG_IMAGES_VIS", save_dir=getattr(self, "work_dir", "./YOLOX_outputs"))
            return mix_img, padded_labels, img_info, img_id

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id = self._dataset.pull_item(idx)
            while len(label) == 0 and self.empty_ignore:
                idx = random.randint(0, self.__len__() - 1)
                img, label, img_info, img_id = self._dataset.pull_item(idx)
            new_label = [poly2obb_np(l) for l in label]
            if len(new_label) == 0:
                new_label = np.zeros((0, 6), dtype=np.float32)
            else:
                new_label = np.concatenate(new_label, axis=0)
                new_label = mintheta_obb(new_label)
            img, label = self.preproc(img, new_label, self.input_dim)
            return img, label, img_info, img_id

    def mixup(self, 
              origin_img, 
              origin_labels, 
              extra_img, 
              extra_labels):
        origin_labels = np.vstack((origin_labels, extra_labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * extra_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels

    def copy_paste(self, 
                   origin_img, 
                   origin_labels, 
                   extra_img,
                   extra_labels,
                   choice_prob=1.):

        cp_img = np.zeros(extra_img.shape, dtype=np.uint8)
        num_targets = len(extra_labels)
        sample_inds = np.asarray(random.sample(range(num_targets), 
                k=max(int(num_targets * choice_prob), 1)))
        
        sample_labels = extra_labels[sample_inds]
        cp_labels = sample_labels.copy()
        for cp_label in cp_labels:
            # iof = bt.bbox_overlaps(origin_labels, cp_label[:-1][None], mode="iof")
            # if iof.all() < 0.5:
            cv2.drawContours(
                cp_img , [cp_label[:-1].reshape(-1, 1, 2).astype(np.int32)], 
                -1, (255, 255, 255), cv2.FILLED)
        # cropped_inst = cv2.bitwise_and(src1=extra_img, src2=cp_img)
        inst_mask = cp_img > 0
        # inst_mask = cropped_inst > 0
        origin_img[inst_mask] = extra_img[inst_mask]
        origin_labels = np.vstack((origin_labels, sample_labels))

        return origin_img, origin_labels

    def _generate_mosaic_image(self, input_dim, indices=None, mosaic_labels=None):
        if mosaic_labels is None: 
            mosaic_labels = []
        if indices is None:
            idx = random.randint(0, self.__len__() - 1)
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]
        input_h, input_w = input_dim[0], input_dim[1]
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
        for i_mosaic, index in enumerate(indices):
            img, _labels, _, img_id = self._dataset.pull_item(index)
            h0, w0 = img.shape[:2]  # orig hw
            scale = min(1. * input_h / h0, 1. * input_w / w0)
            img = cv2.resize(
                img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
            )
            # generate output mosaic image
            (h, w, c) = img.shape[:3]
            if i_mosaic == 0:
                mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

            # suffix l means large image, while s means small image in mosaic aug.
            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
            )

            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            labels = _labels.copy()
            if _labels.size > 0:
                labels[:, 0:8:2] = scale * _labels[:, 0:8:2] + padw
                labels[:, 1:8:2] = scale * _labels[:, 1:8:2] + padh
            mosaic_labels.append(labels)

        if len(mosaic_labels) != 0:
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            mosaic_labels = box_candidates(mosaic_labels, 
                    (2 * input_h, 2 * input_w), (0, 0))
        return mosaic_img, mosaic_labels, img_id
        
    def extra_augmention(self, 
                         origin_img, origin_labels, 
                         input_dim, aug_type="mixup",
                         resample=False, *args, **kwargs):
        assert aug_type in ["copy_paste", "mixup"]
        if resample:
            assert aug_type == "copy_paste", "resample only support copy-paste"
        jit_factor = random.uniform(*self.mixup_scale)
        jit_input_dim = (int(input_dim[0] * jit_factor), int(input_dim[1] * jit_factor))
        FLIP = random.uniform(0, 1) > 0.5
        if resample:
            img, cp_labels, _, _ = self._dataset.resample_pull_item()
        else:
            cp_labels = []
            while len(cp_labels) == 0:
                cp_index = random.randint(0, self.__len__() - 1)
                cp_labels = self._dataset.load_anno(cp_index)
            img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((jit_input_dim[0], jit_input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(jit_input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(jit_input_dim[0] / img.shape[0], jit_input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[0] * cp_scale_ratio), int(img.shape[1] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR
        )
        cp_img[: resized_img.shape[0], : resized_img.shape[1]] = resized_img

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :-1].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )

        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = \
            cp_bboxes_transformed_np[:, 0::2] - x_offset
        cp_bboxes_transformed_np[:, 1::2] = \
            cp_bboxes_transformed_np[:, 1::2] - y_offset
        cp_bboxes_transformed_np, cand_inds = box_candidates(cp_bboxes_transformed_np,
            (target_h - 1, target_w - 1), (-1, -1), True)

        if len(cp_bboxes_transformed_np) == 0:
            return origin_img, origin_labels
        
        cls_labels = cp_labels[cand_inds][..., -1, None]
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        if aug_type == "mixup":
            return self.mixup(origin_img, origin_labels, 
                    padded_cropped_img, labels)
        elif aug_type == "copy_paste":
            return self.copy_paste(origin_img, origin_labels, 
                    padded_cropped_img, labels, **kwargs)
