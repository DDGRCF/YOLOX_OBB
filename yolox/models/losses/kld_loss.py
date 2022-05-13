import torch 
import torch.nn as nn
import numpy as np

def get_sigma(input, eps=1e-7):
    _, wh, theta = input.split([2, 2, 1], -1)
    wh = wh.clamp(min=eps)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    R = torch.cat((Cos, -Sin, Sin, Cos), -1).view(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    sigma = (R @ S.square() @ R.transpose(1, 2)).reshape(-1, 2, 2)
    return sigma

def compute_gwd(pred, target, eps=1e-7, alpha=1.0, tau=1.0, norm=True):
    pred_xy = pred[..., :2]
    target_xy = target[..., :2]
    pred_sigma = get_sigma(pred, eps)
    target_sigma = get_sigma(target, eps)
    # m calculate
    xy_dist = (pred_xy - target_xy).square().sum(-1)
    whr_dist = pred_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_dist = whr_dist + target_sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_tr = (pred_sigma @ target_sigma).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (pred_sigma.det() * target_sigma.det()).clamp(0).sqrt()
    whr_dist = whr_dist + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt()
    )
    dist = (xy_dist + alpha * alpha * whr_dist).clamp(0).sqrt()
    if norm:
        scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(eps)
        dist = dist / scale
    loss = 1 - 1 / (tau + torch.log1p(dist))
    return loss

def compute_kld(pred, target, alpha=1.0, tau=1.0, sqrt=True, eps=1e-7):
    xy_p = pred[..., :2]
    xy_t = target[..., :2]
    Sigma_p = get_sigma(pred)
    Sigma_t = get_sigma(target)
    _shape = xy_p.shape
    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                            -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                            dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(
        dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.bmm(
        Sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = distance.clamp(0).sqrt()

    distance = distance.reshape(_shape[:-1])
    loss = 1 - 1 / (tau + torch.log1p(distance))

    return loss


class KLDLoss(nn.Module):
    def __init__(self, 
                 loss_weight=1.0,
                 reduction='mean',
                 loss_type='kld',
                 eps=1e-6):
        super().__init__()
        self.eps = eps
        self.loss_type = loss_type
        self.reduction = reduction
        self.loss_weight = loss_weight

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self,
                pred,
                target,
                avg_factor=1.,
                **kwargs):
        pred = pred.float()
        target = target.float()
        if self.loss_type == "kld":
            loss = self.loss_weight * compute_kld(pred, target, eps=self.eps, **kwargs)
        elif self.loss_type == "gwd":
            loss = self.loss_weight * compute_gwd(pred, target, eps=self.eps, **kwargs)
        else:
            raise NotImplemented
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise NotImplemented
        return loss / avg_factor
    

