import torch
import numpy as np

from . import nms_rotated_ext
from yolox.utils.obb_utils import (bbox2type, get_bbox_type)

class RotatedNMSOp(torch.autograd.Function):

    @staticmethod
    def symbolic(g, bboxes, scores, iou_threshold, score_threshold, small_threshold, max_num):
        return g.op(
                "yolox::RotatedNonMaxSuppression",
                bboxes,
                scores,
                iou_threshold_f=float(iou_threshold),
                score_threshold_f=float(score_threshold),
                small_thre_f=float(small_threshold),
                max_num_i=int(max_num))

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, bboxes, scores, iou_threshold, score_threshold, small_threshold, max_num):
        assert bboxes.size(0) == scores.size(0)
        assert bboxes.size(1) == 5
        is_score_threshold = score_threshold > 0   
        is_small_threshold = small_threshold > 0
        if is_small_threshold:
            valid_mask1 = bboxes[:, 2:4].min(1)[0] > small_threshold 
            bboxes = bboxes[valid_mask1]
            scores = scores[valid_mask1]
            valid_inds1 = torch.nonzero(valid_mask1, as_tuple=False).squeeze(1)
        if is_score_threshold:
            valid_mask2 = scores > score_threshold 
            bboxes = bboxes[valid_mask2]
            scores = scores[valid_mask2]
            valid_inds2 = torch.nonzero(valid_mask2, as_tuple=False).squeeze(1)
        inds = nms_rotated_ext.nms_rotated(bboxes, scores, iou_threshold)
        if max_num > 0 and len(inds) > max_num:
            inds = inds[:max_num]
        if len(inds):
            if is_score_threshold:
                inds = valid_inds2[inds]
            if is_small_threshold:
                inds = valid_inds1[inds]
        return inds


class PolyNMSOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, bboxes, scores, iou_threshold, score_threshold, small_threshold, max_num):
        return g.op(
                "yolox::PolyNonMaxSuppression",
                bboxes,
                scores,
                iou_threshold_f=float(iou_threshold),
                score_threshold_f=float(score_threshold),
                small_threshold_f=float(small_threshold),
                max_num_i=int(max_num))

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, bboxes, scores, iou_threshold, score_threshold, small_threshold, max_num):
        assert bboxes.size(0) == scores.size(0)
        assert bboxes.size(1) == 5
        is_score_threshold = score_threshold > 0   
        is_small_threshold = small_threshold > 0
        if is_small_threshold:
            valid_mask1 = torch.minimum(bboxes[:, 0::2].max(1)[0] - bboxes[:, 0::2].min(1)[0], 
                                        bboxes[:, 1::2].max(1)[0] - bboxes[:, 1::2].min(1)[0]) > small_threshold
            valid_inds1 = torch.nonzero(valid_mask1, as_tuple=False).squeeze(1)
            bboxes = bboxes[valid_inds1]
            scores = scores[valid_inds1]
        if is_score_threshold:
            valid_mask2 = scores > score_threshold 
            valid_inds2 = torch.nonzero(valid_mask2, as_tuple=False).squeeze(1)
            bboxes = bboxes[valid_inds2]
            scores = scores[valid_inds2]
        inds = nms_rotated_ext.nms_poly(bboxes, scores, iou_threshold)
        if max_num > 0 and len(inds) > max_num:
            inds = inds[:max_num]
        if len(inds):
            if is_score_threshold:
                inds = valid_inds2[inds]
            if is_small_threshold:
                inds = valid_inds1[inds]
        return inds


@torch.cuda.amp.autocast()
def obb_nms(rbboxes, scores, iou_thr, score_thr=-1, small_thr=-1, max_num=-1):
    is_numpy = isinstance(rbboxes, np.ndarray)
    if is_numpy:
        rbboxes = torch.from_numpy(rbboxes)
        scores = torch.from_numpy(scores)
    rbboxes = bbox2type(rbboxes, "obb") 
    scores = scores.squeeze(-1)
    if rbboxes.numel() == 0:
        if is_numpy:
            inds = np.empty((0, ), dtype=np.int64)  
        else:
            inds = rbboxes.new_zeros((0, ), dtype=torch.int64)
        return inds
    else:
        inds = RotatedNMSOp.apply(rbboxes, scores, iou_thr, score_thr, small_thr, max_num)
    if is_numpy:
        inds = inds.numpy()
    return inds


@torch.cuda.amp.autocast()
def poly_nms(rbboxes, scores, iou_thr, score_thr, small_thr=-1, max_num=-1):
    is_numpy = isinstance(rbboxes, np.ndarray)
    if is_numpy:
        rbboxes = torch.from_numpy(rbboxes)
        scores = torch.from_numpy(scores)
    rbboxes = bbox2type(rbboxes, "obb") 
    scores = scores.squeeze(-1)
    if rbboxes.numel() == 0:
        if is_numpy:
            inds = np.empty((0, ), dtype=np.int64)  
        else:
            inds = rbboxes.new_zeros((0, ), dtype=torch.int64)
        return inds
    else:
        inds = PolyNMSOp.apply(rbboxes, scores, iou_thr, score_thr, small_thr, max_num)
    if is_numpy:
        inds = inds.numpy()
    return inds


@torch.cuda.amp.autocast()
def multiclass_obb_nms(rbboxes, 
                       scores, 
                       labels=None, 
                       score_factors=None, 
                       iou_thr=0.1, 
                       score_thr=-1, 
                       small_thr=1e-4, 
                       max_num=-1, 
                       class_agnostic=True,
                       obb_type="obb"):  
    assert type(rbboxes) == type(scores)
    is_numpy = isinstance(rbboxes, np.ndarray)
    if is_numpy:
        rbboxes = torch.from_numpy(rbboxes)
        scores = torch.from_numpy(scores)
        if labels is not None:
            labels = torch.from_numpy(labels)
            labels = labels.float()
    rbboxes = rbboxes.float()
    scores = scores.float()
    if score_factors is not None and isinstance(score_factors, torch.Tensor):
        if is_numpy:
            score_factors = torch.from_numpy(score_factors)
        scores = scores * score_factors
    if len(rbboxes) == 0:
        if is_numpy:
            keep = np.empty((0, ), dtype=np.int64)
        else:
            keep = rbboxes.new_zeros((0, ), dtype=torch.int64)
        return keep
    if class_agnostic:
        rbboxes_for_nms = rbboxes
    else:
        assert type(labels) == type(rbboxes)
        hbboxes = bbox2type(rbboxes, "hbb")
        max_coordinate = hbboxes.max() - hbboxes.min()
        offsets = labels.to(rbboxes) * (max_coordinate + torch.tensor(1).to(rbboxes))
        rbboxes_for_nms = rbboxes.clone()
        _bbox_type = get_bbox_type(rbboxes_for_nms)
        if _bbox_type == "obb":
            rbboxes_for_nms[:, :2] = rbboxes_for_nms[:, :2] + offsets
        elif _bbox_type == "poly":
            rbboxes_for_nms = rbboxes_for_nms + offsets
        else:
            raise NotImplementedError

    if obb_type == "obb":
        keep = obb_nms(rbboxes_for_nms, scores, iou_thr, score_thr, small_thr, max_num)
    elif obb_type == "poly":
        keep = poly_nms(rbboxes_for_nms, scores, iou_thr, score_thr, small_thr, max_num)

    return keep.numpy() if is_numpy else keep
