import torch
import torch.nn as nn
from yolox.utils import (get_bbox_type, bbox2type)
from yolox.ops import convex_sort


def get_bbox_areas(bboxes):
    btype = get_bbox_type(bboxes)
    if btype == 'hbb':
        wh = bboxes[..., 2:] - bboxes[..., :2]
        areas = wh[..., 0] * wh[..., 1]
    elif btype == 'obb':
        areas = bboxes[..., 2] * bboxes[..., 3]
    elif btype == 'poly':
        pts = bboxes.view(*bboxes.size()[:-1], 4, 2)
        roll_pts = torch.roll(pts, 1, dims=-2)
        xyxy = torch.sum(pts[..., 0] * roll_pts[..., 1] -
                         roll_pts[..., 0] * pts[..., 1], dim=-1)
        areas = 0.5 * torch.abs(xyxy)
    else:
        raise ValueError('The type of bboxes is notype')

    return areas


def shoelace(pts):
    roll_pts = torch.roll(pts, 1, dims=-2)
    xyxy = pts[..., 0] * roll_pts[..., 1] - \
           roll_pts[..., 0] * pts[..., 1]
    areas = 0.5 * torch.abs(xyxy.sum(dim=-1))
    return areas


def convex_areas(pts, masks):
    nbs, npts, _ = pts.size()
    index = convex_sort(pts, masks)
    index[index == -1] = npts
    index = index[..., None].repeat(1, 1, 2)

    ext_zeros = pts.new_zeros((nbs, 1, 2))
    ext_pts = torch.cat([pts, ext_zeros], dim=1)
    polys = torch.gather(ext_pts, 1, index)

    xyxy = polys[:, 0:-1, 0] * polys[:, 1:, 1] - \
           polys[:, 0:-1, 1] * polys[:, 1:, 0]
    areas = 0.5 * torch.abs(xyxy.sum(dim=-1))
    return areas


def poly_intersection(pts1, pts2, areas1=None, areas2=None, eps=1e-6):
    # Calculate the intersection points and the mask of whether points is inside the lines.
    # Reference:
    #    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    #    https://github.com/lilanxiao/Rotated_IoU/blob/master/box_intersection_2d.py
    lines1 = torch.cat([pts1, torch.roll(pts1, -1, dims=1)], dim=2)
    lines2 = torch.cat([pts2, torch.roll(pts2, -1, dims=1)], dim=2)
    lines1, lines2 = lines1.unsqueeze(2), lines2.unsqueeze(1)
    x1, y1, x2, y2 = lines1.unbind(dim=-1) # dim: N, 4, 1
    x3, y3, x4, y4 = lines2.unbind(dim=-1) # dim: N, 1, 4

    num = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    den_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    with torch.no_grad():
        den_u = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
        t, u = den_t / num, den_u / num
        mask_t = (t > 0) & (t < 1)
        mask_u = (u > 0) & (u < 1)
        mask_inter = torch.logical_and(mask_t, mask_u)

    t = den_t / (num + eps)
    x_inter = x1 + t * (x2 - x1)
    y_inter = y1 + t * (y2 - y1)
    pts_inter = torch.stack([x_inter, y_inter], dim=-1)

    B = pts1.size(0)
    pts_inter = pts_inter.view(B, -1, 2)
    mask_inter = mask_inter.view(B, -1)

    # Judge if one polygon's vertices are inside another polygon.
    # Use
    with torch.no_grad():
        areas1 = shoelace(pts1) if areas1 is None else areas1
        areas2 = shoelace(pts2) if areas2 is None else areas2

        triangle_areas1 = 0.5 * torch.abs(
            (x3 - x1) * (y4 - y1) - (y3 - y1) * (x4 - x1))
        sum_areas1 = triangle_areas1.sum(dim=-1)
        mask_inside1 = torch.abs(sum_areas1 - areas2[..., None]) < 1e-3 * areas2[..., None]

        triangle_areas2 = 0.5 * torch.abs(
            (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))
        sum_areas2 = triangle_areas2.sum(dim=-2)
        mask_inside2 = torch.abs(sum_areas2 - areas1[..., None]) < 1e-3 * areas1[..., None]

    all_pts = torch.cat([pts_inter, pts1, pts2], dim=1)
    masks = torch.cat([mask_inter, mask_inside1, mask_inside2], dim=1)
    return all_pts, masks


def poly_enclose(pts1, pts2):
    all_pts = torch.cat([pts1, pts2], dim=1)
    mask1 = pts1.new_ones((pts1.size(0), pts1.size(1)))
    mask2 = pts2.new_ones((pts2.size(0), pts2.size(1)))
    masks = torch.cat([mask1, mask2], dim=1)
    return all_pts, masks


def poly_iou_loss(pred, target, mode='linear ', eps=1e-6):
    if pred.size(0) == 0 or target.size(0) == 0:
        return pred.sum() * 0., pred.sum() * 0.
    areas1, areas2 = get_bbox_areas(pred), get_bbox_areas(target)
    pred, target = bbox2type(pred, 'poly'), bbox2type(target, 'poly')
    pred_pts = pred.reshape(pred.size(0), -1, 2)
    target_pts = target.reshape(target.size(0), -1, 2)
    inter_pts, inter_masks = poly_intersection(
        pred_pts, target_pts, areas1, areas2, eps)
    overlap = convex_areas(inter_pts, inter_masks)

    ious = (overlap / (areas1 + areas2 - overlap + eps)).clamp(min=eps, max=1.0)
    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'mve':
        loss =  1 - ious ** 2
    elif mode == 'log':
        loss = -ious.log()
    return loss, ious


def poly_giou_loss(pred, target, eps=1e-6):
    if pred.size(0) == 0 or target.size(0) == 0:
        return pred.sum() * 0., pred.sum() * 0.
    areas1, areas2 = get_bbox_areas(pred), get_bbox_areas(target)
    pred, target = bbox2type(pred, 'poly'), bbox2type(target, 'poly')

    pred_pts = pred.reshape(pred.size(0), -1, 2)
    target_pts = target.view(target.size(0), -1, 2)
    inter_pts, inter_masks = poly_intersection(
        pred_pts, target_pts, areas1, areas2, eps)
    overlap = convex_areas(inter_pts, inter_masks)

    union = areas1 + areas2 - overlap + eps
    ious = (overlap / union).clamp(min=eps)

    enclose_pts, enclose_masks = poly_enclose(pred_pts, target_pts)
    enclose_areas = convex_areas(enclose_pts, enclose_masks)

    gious = ious - (enclose_areas - union) / enclose_areas
    loss = 1 - gious
    return loss, gious


class PolyIoULoss(nn.Module):

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 loss_type='linear',
                 eps=1e-6):
        super().__init__()
        self.loss_type=loss_type
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self,
                pred,
                target,
                avg_factor=1.,
                weight=None,
                return_iou=False,
                **kwargs):
        pred = pred.float()
        target = target.float()
        if weight is not None and isinstance(weight, torch.Tensor):
            weight = weight.float()
        loss, iou = poly_iou_loss(
            pred,
            target,
            mode=self.loss_type,
            eps=self.eps)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        if return_iou:
            return self.loss_weight * loss / avg_factor, iou
        else:
            return self.loss_weight * loss / avg_factor


class PolyGIOULoss(nn.Module):

    def __init__(self,
                 loss_weight=1.0,
                 reduction='none',
                 loss_type="poly",
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        assert reduction in [None, 'none', 'mean', 'sum']
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.kwargs =kwargs

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self,
                pred,
                target,
                avg_factor=1.,
                weight=None,
                return_iou=False,
                **kwargs):
        pred = pred.float()
        target = target.float()
        if weight is not None and isinstance(weight, torch.Tensor):
            weight = weight.float()
            
        if self.loss_type == "poly":
            loss, giou = poly_giou_loss(
                pred,
                target,
                weight,
                eps=self.eps,
                **kwargs)
        else:
            raise NotImplementedError
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        if return_iou:
            return self.loss_weight * loss / avg_factor, giou
        else:
            return self.loss_weight * loss / avg_factor
