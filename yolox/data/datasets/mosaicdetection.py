#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# 禁用远程端口 # TODO: delete
import matplotlib
matplotlib.use("Agg")

import random
import shutil 

import os
import cv2
import numpy as np

import BboxToolkit as bt
from yolox.utils import (adjust_box_anns, get_local_rank, 
                         mintheta_obb, obb2poly, poly2obb_np, bbox2type)
from yolox.utils.mask_utils import resize_mask, mask_overlaps
from ..data_augment import (obox_candidates, random_affine, obb_random_affine, mask_random_affine)
from .datasets_wrapper import Dataset, MaskDataset
from yolox.utils.visualize import _COLORS

def debug_obb_data(labels, 
                   img_o, 
                   save_name=None, 
                   minAreaRect_test=False, 
                   dir_name="DEBUG_IMAGES_VIS", 
                   save_dir=None):
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
    os.makedirs(save_name_dir, exist_ok=True)
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
    return 0

def debug_mask_data(labels, 
               img, 
               mask=None, 
               class_names=None,
               save_name=None, 
               save_root=None, 
               dir_name="DEBUG_IMAGES_VIS",
               bbox_type="xyxy",
               enable_txt=False):
    assert bbox_type in ["xyxy", "cxcywh"]
    if len(labels) == 0: 
        return 0
    if img.shape[-1] not in [1, 3] and img.ndim == 3:
        img = img.transpose((1, 2, 0))
        img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.dtype != np.uint8:
        img = np.ascontiguousarray(img, dtype=np.uint8)
    if save_name is None:
        save_name = '0'
    else:
        if isinstance(save_name, int):
            save_name = str(save_name)
    if save_root is None:
        save_root = "YOLOX_outputs"
    save_name_dir = os.path.join(save_root, dir_name)
    if not os.path.exists(save_name_dir):
        os.makedirs(save_name_dir)
    bboxes = labels[..., 1:]
    if bbox_type == "cxcywh":
        cxcy = bboxes[..., :2]
        wh = bboxes[..., 2:]
        x1y1 = cxcy - wh / 2
        x2y2 = cxcy + wh / 2
        bboxes = np.concatenate((x1y1, x2y2), axis=-1)
    elif bbox_type == "xyxy":
        pass
    else:
        raise NotImplemented

    clses = labels[..., 0]
    if mask is not None:
        if isinstance(mask, list):
            new_mask = []
            for m in mask:
                if m.dtype != np.uint8:
                    new_mask.append(np.ascontiguousarray(m, dtype=np.uint8))
                new_mask.append(m)
            mask = new_mask
    for i, (bbox, cls_id) in enumerate(zip(bboxes, clses)):
        x0 = int(bbox[0]) 
        y0 = int(bbox[1])
        x1 = int(bbox[2])
        y1 = int(bbox[3])
        cls_id = int(cls_id)
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}'.format(class_names[cls_id] if class_names is not None else cls_id)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.8).astype(np.uint8).tolist()
        if enable_txt:
            cv2.rectangle(
                img,
                (x0, y0+1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        # if mask is not None:
        #     if i == 0:
        #         mask_temp = mask.copy()
        #     mask_temp = cv2.rectangle(mask_temp, (x0, y0), (x1, y1), color, 2)
        if np.sum(bbox) == 0:
            break
    img = img.astype(np.uint8)
    assert img.shape[-1] == 3, "img shape is {}".format(img.shape[-1])
    num_targets = mask.shape[0]
    for i in range(num_targets):
        rand_id = np.random.randint(low=0, high=79)
        color = (_COLORS[rand_id] * 255).astype(np.uint8).tolist()
        if img.ndim == 2:
            m_expand = np.astype(color[0], dtype=np.uint8)[None, None] * mask[..., None]
        else:
            m_expand = np.tile(mask[i][..., None], (1, 1, 3)) \
                * np.asarray(color, dtype=np.uint8)[None, None, :]
        assert m_expand.shape[-1] == 3, "mask shape is {}".format(m_expand.shape[-1])
        m_expand = m_expand.astype(np.uint8)
        cv2.addWeighted(img, 1.0, m_expand, 0.7, 0., dst=img)
    img_save_path = os.path.join(save_name_dir, save_name + '.jpg')
    cv2.imwrite(img_save_path, img)
    return 1


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
        mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True, 
        mosaic_prob=1.0, mixup_prob=1.0, 
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
        mixup_scale=(0.5, 1.5), shear=2.0, overlaps_thre=0.6,
        enable_resample=False, mosaic_prob=1.0, mixup_prob=1.0, 
        copy_paste_prob=1.0, enable_debug=False, aug_ignore=None, 
        empty_ignore=True, **kwargs
    ):
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob
        self.enable_debug = enable_debug
        self.empty_ignore = empty_ignore
        self.enable_resample = enable_resample
        self.empty_ignore = empty_ignore
        self.overlaps_thre = overlaps_thre
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
        if self.enable_mosaic:
            AUG_IGNORE_FLAG=False
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]
            if random.random() < self.mosaic_prob:
                indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]
                mosaic_img, mosaic_labels, img_id = self._generate_mosaic_image(input_dim, indices)
                while len(mosaic_labels) == 0 and self.empty_ignore:
                    mosaic_img, mosaic_labels, img_id = self._generate_mosaic_image(input_dim)
                if self.aug_ignore_list is not None:
                    AUG_IGNORE_FLAG = len(np.intersect1d(self.aug_ignore_list, mosaic_labels[..., -1])) > 0.
            else:
                img, _labels, _, img_id = self._dataset.pull_item(idx)
                while len(_labels) == 0 and self.empty_ignore:
                    idx = random.randint(0, self.__len__() - 1)
                    img, _labels, _, img_id = self._dataset.pull_item(idx)
                h0, w0 = img.shape[:2]
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                labels = _labels.copy()
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0]
                    labels[:, 1] = scale * _labels[:, 1]
                    labels[:, 2] = scale * _labels[:, 2]
                    labels[:, 3] = scale * _labels[:, 3]
                mosaic_img = img
                mosaic_labels = labels
                if self.aug_ignore_list is not None:
                    AUG_IGNORE_FLAG = len(np.intersect1d(self.aug_ignore_list, mosaic_labels[..., -1])) > 0.

            mosaic_img, mosaic_labels = obb_random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees if not AUG_IGNORE_FLAG else 0.,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear if not AUG_IGNORE_FLAG else 0.,
            ) 
            if (
                not len(mosaic_labels) == 0
                and random.random() < self.copy_paste_prob
            ):
                mosaic_img, mosaic_labels = self.extra_augmention(
                    mosaic_img, mosaic_labels, self.input_dim, "copy_paste",
                    resample=self.enable_resample, choice_prob=1.0)

            if (
                not len(mosaic_labels) == 0
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
                debug_obb_data(padded_labels, mix_img, str(idx.item()) if isinstance(idx, np.ndarray) else str(idx), 
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
                   choice_prob=1.,
                   overlaps_thre=0.4):

        cp_img = np.zeros(extra_img.shape, dtype=np.uint8)
        num_targets = len(extra_labels)
        sample_inds = np.asarray(random.sample(range(num_targets), 
                k=max(int(num_targets * choice_prob), 1)))
        
        sample_labels = extra_labels[sample_inds]
        cp_labels = sample_labels.copy()
        keep_labels = []
        for cp_label in cp_labels:
            iof = bt.bbox_overlaps(origin_labels[:, :-1], cp_label[:-1][None], mode="iof", is_aligned=False)
            if (iof < overlaps_thre).all():
                keep_labels.append(cp_label)
                cv2.drawContours(
                    cp_img , [cp_label[:-1].reshape(-1, 1, 2).astype(np.int32)], 
                    -1, (255, 255, 255), cv2.FILLED)
        inst_mask = cp_img > 0
        origin_img[inst_mask] = extra_img[inst_mask]
        if len(keep_labels):
            keep_labels = np.stack(keep_labels, axis=0)
            origin_labels = np.vstack((origin_labels, keep_labels))

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
            _, valid_inds = obox_candidates(mosaic_labels[:, :8], 
                    (2 * input_h, 2 * input_w), (0, 0), self.overlaps_thre, True)
            mosaic_labels = mosaic_labels[valid_inds]
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
        cp_bboxes_transformed_np, cand_inds = obox_candidates(cp_bboxes_transformed_np,
            (target_h - 1, target_w - 1), (-1, -1), self.overlaps_thre, True)

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


class MaskAugDataset(MaskDataset):
    def __init__(
        self, dataset, img_size, augmention=True, preproc=None,
        degrees=10.0, translate=0.1, scale=(0.8, 1.2),
        shear=2.0, mosaic_prob=1.0, 
        enable_debug=False, copy_paste_prob=0.0,
        *args, **kwargs
    ):
        super().__init__(img_size, augmention=augmention)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.mosaic_prob = mosaic_prob
        self.copy_paste_prob = copy_paste_prob

        self.enable_augmention = augmention
        self.enable_debug = enable_debug
        self.local_rank = get_local_rank()

    def __len__(self):
        return len(self._dataset)
    
    @Dataset.wrapper_getitem
    def __getitem__(self, idx):
        if self.enable_augmention:
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]
            if random.random() < self.mosaic_prob:
                mosaic_labels = []
                # yc, xc = s, s  # mosaic center x, y
                yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
                xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

                # 3 additional image indices
                indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

                for i_mosaic, index in enumerate(indices):
                    img, _labels, _, img_id, masks = self._dataset.pull_item(index)
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

                    # mask_deals
                    num_targets = len(_labels)
                    if i_mosaic == 0:
                        mosaic_masks = []
                    if num_targets:
                        masks = resize_mask(
                            masks, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
                        )
                        mosaic_mask = np.full((input_h * 2, input_w * 2, num_targets), 0, dtype=np.uint8)
                        mosaic_mask[l_y1:l_y2, l_x1:l_x2] = masks[s_y1:s_y2, s_x1:s_x2]
                        mosaic_masks.append(mosaic_mask)

                    labels = _labels.copy()
                    # Normalized xywh to pixel xyxy format
                    if _labels.size > 0:
                        labels[:, 0] = scale * _labels[:, 0] + padw
                        labels[:, 1] = scale * _labels[:, 1] + padh
                        labels[:, 2] = scale * _labels[:, 2] + padw
                        labels[:, 3] = scale * _labels[:, 3] + padh

                    mosaic_labels.append(labels)

                mosaic_masks = np.concatenate(mosaic_masks, axis=-1) # (2 * input_h, 2 * input_w, num_targets)
                    
                if len(mosaic_labels):
                    mosaic_labels = np.concatenate(mosaic_labels, 0)
                    np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                    np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                    np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                    np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
            else:
                img, _labels, _, img_id, masks = self._dataset.pull_item(idx)
                h0, w0 = img.shape[:2]
                scale = min(1. * input_h / h0, 1 * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                (h, w, c) = img.shape[:3]
                mosaic_img = np.full((input_h, input_w, c), 114, dtype=np.uint8)
                mosaic_img[:h, :w, :] = img
                num_targets = len(_labels)
                mosaic_masks = np.full((input_h, input_w, num_targets), 0, dtype=np.uint8)
                if num_targets:
                    masks = resize_mask(masks.copy(), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                    mosaic_masks[:h, :w] = masks
                mosaic_labels = _labels.copy()

            mosaic_img, mosaic_labels, mosaic_masks = mask_random_affine(
                mosaic_img,
                mosaic_labels,
                masks=mosaic_masks,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
            )

            if random.random() < self.copy_paste_prob:
                mosaic_img, mosaic_labels, mosaic_masks = self.copy_paste(
                    mosaic_img, mosaic_labels, mosaic_masks, self.input_dim
                )

            mosaic_img, mosaic_labels, mosaic_masks = self.preproc(
                mosaic_img, mosaic_labels, self.input_dim, mask_targets=mosaic_masks)

            if self.enable_debug and mosaic_labels.sum() > 0:
                valid_masks = (mosaic_masks.sum((-2, -1)) > 0)
                assert valid_masks.all(), f"{valid_masks}"
                debug_mask_data(mosaic_labels.copy(), 
                           mosaic_img.copy(), 
                           mosaic_masks.copy(), 
                           class_names=self._dataset._classes, 
                           save_name=idx.item(), 
                           bbox_type='cxcywh')

            img_info = (mosaic_img.shape[1], mosaic_img.shape[0])
            return mosaic_img, mosaic_labels, mosaic_masks, img_info, img_id

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id, mask = self._dataset.pull_item(idx)
            img, label, mask = self.preproc(img, label, self.input_dim, mask)
            return img, label, mask, img_info, img_id
        
    def copy_paste(self, origin_img, origin_labels, origin_masks, input_dim, iof_thre=0.5):
        jit_factor = random.uniform(*self.scale)
        jit_input_dim = (int(input_dim[0] * jit_factor), int(input_dim[1] * jit_factor))
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _, masks = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((jit_input_dim[0], jit_input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(jit_input_dim, dtype=np.uint8) * 114
        
        cp_scale_ratio = min(jit_input_dim[0] / img.shape[0], jit_input_dim[1] / img.shape[1])

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
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

        num_targets = len(cp_labels)
        cp_masks = np.zeros(
            (jit_input_dim[0], jit_input_dim[1], num_targets), dtype=np.uint8
        )
        resized_masks = resize_mask(
            masks, (int(masks.shape[1] * cp_scale_ratio), int(masks.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR
        )
        cp_masks[: resized_masks.shape[0], : resized_masks.shape[1]] = resized_masks

        if FLIP:
            cp_masks = cp_masks[:, ::-1, :]
        padded_masks = np.zeros((padded_img.shape[0], padded_img.shape[1], num_targets), dtype=np.uint8)
        padded_masks[:origin_h, :origin_w] = cp_masks

        padded_cropped_masks = padded_masks[
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

        cls_labels = cp_labels[:, 4:].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        mask_iof = mask_overlaps(origin_masks, padded_cropped_masks, n_first=False, is_aligned=False, mode="iof") # (n1, n2)
        empty_img = np.zeros(padded_cropped_img.shape, dtype=np.uint8)
        keep_labels = []
        keep_masks = []
        for i, label in enumerate(labels):
            if (mask_iof[:, i] < iof_thre).all():
                keep_labels.append(label)
                keep_masks.append(padded_cropped_masks[..., i])
                empty_img += padded_cropped_masks[..., [i]]
                # cv2.drawContours(
                #     empty_img , [label[:-1].reshape(-1, 1, 2).astype(np.int32)], 
                #     -1, (255, 255, 255), cv2.FILLED)
        inst_mask = empty_img > 0
        origin_img[inst_mask] = padded_cropped_img[inst_mask]
        if len(keep_labels):
            keep_labels = np.stack(keep_labels, axis=0)
            keep_masks = np.stack(keep_masks, axis=-1)
            origin_labels = np.vstack((origin_labels, keep_labels))
            # origin_img = origin_img.astype(np.float32)
            # origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)
            origin_masks = np.concatenate((origin_masks, keep_masks), axis=-1, dtype=np.uint8)


        return origin_img.astype(np.uint8), origin_labels, origin_masks

