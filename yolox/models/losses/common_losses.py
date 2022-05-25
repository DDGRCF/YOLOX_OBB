import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        loss = loss * weight
    return loss

class L1Loss(nn.Module):
    def __init__(self, 
                 loss_weight=1., 
                 reduction='none', 
                 loss_type="norm",
                 **kwargs):
        super(L1Loss, self).__init__()
        assert reduction in [ None, 'NONE', 'None','none', 'mean', 'sum']
        assert loss_type in ['norm', 'smooth']
        self.reduction = reduction
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.kwargs = kwargs
    
    def forward(self, pred, target, avg_factor=1.):
        assert pred.shape[0] == target.shape[0], \
            f"expect {pred.shape} == {target.shape}"
        if pred.shape[0] == 0:
            return pred.sum() * 0.
        if self.loss_type == "norm":
            loss = F.l1_loss(pred, target, reduction="none", **self.kwargs)
        elif self.loss_type == "smooth":
            loss = F.smooth_l1_loss(pred, target, reduction="none", **self.kwargs)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        return self.loss_weight * loss / avg_factor


class IoULoss(nn.Module):
    def __init__(self, 
                 loss_weight=1.,
                 reduction="none",
                 loss_type="iou",
                 return_iou=False):
        super(IoULoss, self).__init__()
        assert reduction in [None, 'none', 'mean', 'sum']
        assert loss_type in ["iou", "giou"]
        self.reduction = reduction
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.return_iou = return_iou

    def forward(self, pred, target, avg_factor=1.):
        # pred: xywh, xywh
        assert pred.shape[0] == target.shape[0], \
            f"expect {pred.shape} == {target.shape}"
        if pred.shape[0] == 0:
            return pred.sum() * 0.
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        if self.return_iou:
            return self.loss_weight * loss / avg_factor, giou if self.loss_type=="giou" else iou
        else:
            return self.loss_weight * loss / avg_factor

    
class DiceLoss(nn.Module):
    def __init__(self, 
                 loss_weight=1., 
                 reduction="none", 
                 loss_type='sqrt', 
                 eps=1e-5):
        super(DiceLoss, self).__init__()
        assert reduction in [ None, 'none', 'mean', 'sum']
        assert loss_type in ["norm", "sqrt"]
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.eps = eps

    def forward(self, pred, target, avg_factor=1.):
        assert pred.shape[0] == target.shape[0], \
            f"expect {pred.shape} == {target.shape}"
        if pred.shape[0] == 0:
            return pred.sum() * 0.
        num_pred = pred.shape[0]
        pred = pred.view(num_pred, -1)
        target = target.view(num_pred, -1)
        if self.loss_type == "sqrt":
            intersection = (pred * target).sum(1)
            union = (pred.pow(2)).sum(1) \
                + (target.pow(2)).sum(1) + self.eps
            loss = 1. - (2. * intersection / union)
        elif self.loss_type == "norm":
            intersection = 2 * (pred * target).sum(1)
            union = pred.sum(1) + target.sum(1) + self.eps
            loss = 1 - intersection / union

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return self.loss_weight * loss / avg_factor


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.,
                 reduction="none",
                 loss_type="bce_use_sigmoid",
                 **kwargs):
        super(CrossEntropyLoss, self).__init__()
        assert reduction in [ None, 'none', 'mean', 'sum']
        assert loss_type in ["ce", "bce", "bce_use_sigmoid"]
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.kwargs = kwargs

    def forward(self, pred, target, avg_factor=1.):
        assert pred.shape[0] == target.shape[0], \
            f"expect {pred.shape} == {target.shape}"
        if pred.shape[0] == 0:
            return pred.sum() * 0.
        if self.loss_type == "bce_use_sigmoid":
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none", **self.kwargs)
        elif self.loss_type == "bce":
            loss = F.binary_cross_entropy(pred, target, reduction="none", **self.kwargs)
        elif self.loss_type == "ce":
            loss = F.cross_entropy(pred, target, reduction="none", **self.kwargs)
        else:
            raise NotImplementedError
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        return self.loss_weight * loss / avg_factor


class FocalLoss(nn.Module):

    def __init__(self,
                 loss_weight=1.,
                 reduction="none",
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=1.):
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha)
        else:
            raise NotImplementedError
        if self.reduction == "mean":
            loss_cls = loss_cls.mean()
        elif self.reduction == "sum":
            loss_cls = loss_cls.sum()
        return loss_cls / avg_factor