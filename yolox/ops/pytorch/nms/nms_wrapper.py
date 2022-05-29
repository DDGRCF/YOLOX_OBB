import torch
import torchvision
import numpy as np

class NMSOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bboxes, scores, iou_threshold, score_threshold,
                max_num):
        is_filtering_by_score = score_threshold > 0
        if is_filtering_by_score:
            valid_mask = (scores > score_threshold).squeeze(-1)
            bboxes, scores = bboxes[valid_mask], scores[valid_mask]
            valid_inds = torch.nonzero(
                valid_mask, as_tuple=False).squeeze(dim=1)

        inds = torchvision.ops.nms(
            bboxes, scores, iou_threshold=float(iou_threshold))
        if max_num > 0:
            inds = inds[:max_num]
        if is_filtering_by_score:
            inds = valid_inds[inds]
        return inds

    @staticmethod
    def symbolic(g, bboxes, scores, iou_threshold, score_threshold,
                 max_num):
        from torch.onnx.symbolic_opset9 import select, squeeze, unsqueeze

        from ..onnx_utils  import _size_helper

        boxes = unsqueeze(g, bboxes, 0) # (1, n, 4)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0) # (1, 1, n)

        if max_num > 0:
            max_num = g.op(
                'Constant',
                value_t=torch.tensor(max_num, dtype=torch.long))
        else:
            dim = g.op('Constant', value_t=torch.tensor(0))
            max_num = _size_helper(g, bboxes, dim)
        max_output_per_class = max_num
        iou_threshold = g.op(
            'Constant',
            value_t=torch.tensor([iou_threshold], dtype=torch.float))
        score_threshold = g.op(
            'Constant',
            value_t=torch.tensor([score_threshold], dtype=torch.float))
        nms_out = g.op('NonMaxSuppression', boxes, scores,
                        max_output_per_class, iou_threshold,
                        score_threshold)
        return squeeze(
            g,
            select(
                g, nms_out, 1,
                g.op(
                    'Constant',
                    value_t=torch.tensor([2], dtype=torch.long))), 1)
    
def multiclass_nms(bboxes, 
                   scores, 
                   labels=None, 
                   score_factors=None,
                   iou_thr=0.1,
                   score_thr=-1,
                   max_num=-1,
                   class_agnostic=True): 
    is_numpy = isinstance(bboxes, np.ndarray)
    if is_numpy:
        bboxes = torch.from_numpy(bboxes)
        scores = torch.from_numpy(scores)
        if labels is not None:
            labels = torch.from_numpy(labels)
            labels = labels.float()
    # defalut to set float type
    bboxes = bboxes.float()
    scores = scores.float()
    if score_factors is not None and isinstance(score_factors, torch.Tensor):
        if is_numpy:
            score_factors = torch.from_numpy(score_factors)
        scores = scores * score_factors
    if len(bboxes) == 0:
        if is_numpy:
            keep = np.empty((0, ), dtype=np.int64)
        else:
            keep = bboxes.new_zeros((0, ), dtype=torch.int64)
        return keep
    if class_agnostic:
        bboxes_for_nms = bboxes
    else:
        max_coordinate = bboxes.max() - bboxes.min()
        offsets = labels.to(bboxes) * (max_coordinate + torch.tensor(1).to(bboxes))
        bboxes_for_nms = bboxes.clone()
        # bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
        bboxes_for_nms = bboxes_for_nms + offsets[:, None]
    # if torch.jit.is_tracing():
    #     is_filtering_by_score = score_thr > 0
    #     if is_filtering_by_score:
    #         valid_mask = scores > score_thr
    #         bboxes, scores = bboxes[valid_mask], scores[valid_mask]
    #         valid_inds = torch.nonzero(
    #             valid_mask, as_tuple=False).squeeze(dim=1)

    #     inds = torchvision.ops.nms(
    #         bboxes, scores, iou_threshold=float(iou_thr))

    #     if max_num > 0:
    #         inds = inds[:max_num]
    #     if is_filtering_by_score:
    #         inds = valid_inds[inds]
    #     keep = inds
    # else:
    keep = NMSOp.apply(bboxes_for_nms, scores, iou_thr, score_thr, max_num) 

    return keep.numpy() if is_numpy else keep