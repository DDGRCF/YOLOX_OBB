import math
from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.nn import init
from .detect import Detect
from ..init_functions import *
from yolox.utils import get_world_size, is_dist_avail_and_initialized, mask_overlaps, dice_score

class SparseInstDetect(Detect):
    def __init__(self, 
                 in_channels=(256, 128),
                 num_classes=80,
                 num_masks=100,
                 scale_factor=2,
                 alpha=0.8, beta=0.2,
                 inplace=True):
        super().__init__()
        self.num_classes = num_classes
        self.inst_dim = in_channels[0]
        self.kernel_dim = in_channels[1]
        self.num_masks = num_masks
        self.inplace = inplace

        self.strides = 8
        self.iam_conv = nn.Conv2d(self.inst_dim, self.num_masks, 3, 1, 1)
        self.cls_score = nn.Linear(self.inst_dim, self.num_classes)
        self.mask_kernel = nn.Linear(self.inst_dim, self.kernel_dim)
        self.objectness = nn.Linear(self.inst_dim, 1)
        self.scale_factor = scale_factor
        self.alpha = alpha
        self.beta = beta
        self.mask_score = dice_score
        # self.mask_stream = torch.cuda.Stream()

    def initialize_biases(self):
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_init_with_prob(self.prior_prob))
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)
        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

    def _forward(self, xin):
        inst_features = xin[0]
        mask_features = xin[1]
        iam = self.iam_conv(inst_features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = inst_features.size(1) # 256
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob, inst_features.view(B, C, -1).permute(0, 2, 1)) # (BXNXH*W) * (BXH*WXC) -> (BXNXC)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        inst_features = inst_features / normalizer[:, :, None] # (BXNXC) / (BXN) -> (BXNXC)
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features) # (bs, n, c) -> (bs, n, nc)
        pred_kernel = self.mask_kernel(inst_features) # (bs, n, c) -> (bs, n, 128)
        pred_scores = self.objectness(inst_features) # (bs, n, c) -> (bs, n, 1)

        N = pred_kernel.shape[1]
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, mask_features.view(B, C, H * W)).view(B, N, H, W)

        pred_masks = F.interpolate(
            pred_masks, scale_factor=self.scale_factor,
            mode='bilinear', align_corners=False)

        return pred_logits, pred_masks, pred_scores
    
    def forward(self, xin):
        outputs = self._forward(xin)
        if self.training:
            return outputs
        else:
            pass
    
    def get_losses(self, targets, inps):
        # targets
        device = inps[0].device
        dtype = inps[0].dtype
        masks_t = targets[1] # (list)
        if not hasattr(self, "mask_stream"):
            self.mask_stream = torch.cuda.Stream()

        with torch.cuda.stream(self.mask_stream):
            target_masks = [torch.from_numpy(m).to(device, non_blocking=True).type(dtype) for m in masks_t]
        targets = [t for t in targets[0]]
        # nlabel = (targets.sum(dim=2) > 0).sum(dim=1) # (bs)
        nlabel = [(t.sum(dim=1)>0).sum(dim=0).item() for t in targets]
        labels_t = [bbox_target[:n, 0] for n, bbox_target in zip(nlabel, targets)]
        bboxes_t = [bbox_target[:n, 1:4] for n, bbox_target in zip(nlabel, targets)]
        target_labels = torch.cat(labels_t, dim=0)
        target_bboxes = torch.cat(bboxes_t, dim=0)
        # preds
        pred_logits, pred_masks, pred_scores = inps
        nlabels_t = torch.as_tensor(
            [sum(nlabel)], dtype=torch.float, device=pred_logits.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(nlabels_t)
        nlabels_t = torch.clamp(nlabels_t / get_world_size(), min=1).item()

        torch.cuda.current_stream().wait_stream(self.mask_stream)
        for target_mask in target_masks:
            target_mask.record_stream(torch.cuda.current_stream())
        target_masks = torch.cat(target_masks, dim=0)
        target_masks = F.interpolate(
            target_masks[:, None], size=pred_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        indices = self.sparseinst_matcher(target_masks, target_bboxes, target_labels, pred_masks, pred_logits, nlabel)
        # logits loss
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(labels_t, indices)]).type(torch.long)
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o

        pred_logits = pred_logits.flatten(0, 1)
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(pred_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        # comp focal loss.
        loss_class = self.class_loss(
            pred_logits,
            labels, 
            avg_factor=nlabels_t)

        # dice and ce loss
        src_idx = idx
        tgt_idx = self._get_tgt_permutation_idx(indices)
        if len(target_masks) == 0:
            loss_dice = pred_masks.sum() * 0.
            loss_bce = pred_masks.sum() * 0.
            loss_obj = pred_scores.sum() * 0.
        else:
            pred_masks = pred_masks[src_idx]
            mix_tgt_idx = torch.zeros_like(tgt_idx[1])
            cum_sum = 0
            for num_mask in nlabel:
                mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
                cum_sum += num_mask
            mix_tgt_idx += tgt_idx[1]

            target_masks = target_masks[mix_tgt_idx]

            with torch.no_grad():
                ious = mask_overlaps(pred_masks, target_masks, threshold=[0.4, 0.5], dtype=dtype)

            tgt_iou_scores = ious
            src_iou_scores = pred_scores[src_idx]
            tgt_iou_scores = tgt_iou_scores.flatten(0)
            src_iou_scores = src_iou_scores.flatten(0)
            pred_masks = pred_masks.flatten(1)
            target_masks = target_masks.flatten(1)

            loss_dice = self.dice_loss(
                pred_masks, 
                target_masks, 
                avg_factor=nlabels_t
            )
            loss_bce = self.bce_loss(
                pred_masks,
                target_masks,
                avg_factor=nlabels_t
            )
            loss_obj = self.obj_loss(
                src_iou_scores, 
                tgt_iou_scores
            )

            losses = dict(
                loss_class = loss_class,
                loss_dice = loss_dice,
                loss_bce = loss_bce,
                loss_obj = loss_obj
            )
            return losses
        
    @torch.no_grad()
    def sparseinst_matcher(self, masks_t, bboxes_t, labels_t, pred_masks, pred_logits, nlabel):
        B, N, H, W = pred_masks.shape
        # tgt_ids = torch.cat([v["labels"] for v in targets]) # classes id
        pred_logits = pred_logits.sigmoid()
        tgt_masks = masks_t
        tgt_ids = labels_t.long()
        if tgt_ids.shape[0] == 0:
            return [(torch.as_tensor([]).to(pred_logits), torch.as_tensor([]).to(pred_logits))] * B
        # device = pred_masks.device
        # tgt_masks = tgt_masks.to(pred_masks)
        # tgt_masks = F.interpolate(
        #     tgt_masks[:, None], size=pred_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

        pred_masks = pred_masks.view(B * N, -1)
        tgt_masks = tgt_masks.flatten(1)

        mask_score = self.mask_score(pred_masks, tgt_masks) # (bs * N, gts)
        # Nx(Number of gts)
        matching_prob = pred_logits.view(B * N, -1)[:, tgt_ids]
        C = (mask_score ** self.alpha) * (matching_prob ** self.beta)
        C = C.view(B, N, -1).cpu() # (bs, N, gts)
        # hungarian matching
        # sizes = [n.item() for n in nlabel]
        indices = [linear_sum_assignment(c[i], maximize=True)
                    for i, c in enumerate(C.split(nlabel, -1))] # [(N_id, gts_id)]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(
            j, dtype=torch.int64)) for i, j in indices]
        return indices

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx