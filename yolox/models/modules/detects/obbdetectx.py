import torch
import torch.nn.functional as F
from .detectx import DetectX
from yolox.utils import obbpostprocess
from yolox.ops import OBBOverlaps

class OBBDetectX(DetectX):
    def __init__(
                 self, 
                 anchors=1, 
                 in_channels=(
                     128, 128, 128, 128, 128, 128),
                 reg_dim=5,
                 num_classes=15, 
                 **kwargs):
        super().__init__(
            anchors, 
            in_channels,
            reg_dim=reg_dim,
            num_classes=num_classes,
            **kwargs)
        self.cal_overlaps = OBBOverlaps()
    
    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_rbboxes_per_image,
        gt_classes,
        reg_preds_per_image,
        expanded_strides,
        xy_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):
        
        # gt_bboxes_per_image:shape(n_gt, 5) bboxes_preds_per_image:shape(a_anchors, 5)
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_rbboxes_per_image = gt_rbboxes_per_image.cpu().float()
            reg_preds_per_image = reg_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            xy_shifts = xy_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_rbboxes_per_image,
            expanded_strides,
            xy_shifts,
            num_gt,
        )
        assert fg_mask.any()
        reg_preds_per_image = reg_preds_per_image[fg_mask] # shape(num_in, 5)
        cls_preds_ = cls_preds[batch_idx][fg_mask] # shape(num_in, num_classes)
        obj_preds_ = obj_preds[batch_idx][fg_mask] # shape(num_in, 1)
        num_in_boxes_anchor = reg_preds_per_image.shape[0]

        if mode == "cpu":
            gt_rbboxes_per_image = gt_rbboxes_per_image.cpu()
            reg_preds_per_image = reg_preds_per_image.cpu()
        pair_wise_ious = self.cal_overlaps(gt_rbboxes_per_image, reg_preds_per_image)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        ) # shape(num_gt, num_in, 1)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).sigmoid_()
                * obj_preds_.float().unsqueeze(0).sigmoid_()
            ).repeat(num_gt, 1, 1)
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1) 
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        ) 

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_rbboxes_per_image,
        expanded_strides,
        xy_shifts,
        num_gt,
    ):
        gt_angles_per_image = gt_rbboxes_per_image[:, 4, None] # shape(num_gt, 1)
        gt_xy_per_image = gt_rbboxes_per_image[:, None, 0:2] # shape(num_gt, 1, 2)
        gt_wh_per_image = gt_rbboxes_per_image[:, None, 2:4] # shape(num_gt, 1, 2)
        expanded_strides_per_image = expanded_strides[..., None]
        xy_shifts_per_image = xy_shifts * expanded_strides_per_image
        grid_xy_per_image = xy_shifts_per_image + 0.5 * expanded_strides_per_image # shape(1, n_anchor, 2)
       
        total_num_anchors = grid_xy_per_image.shape[1] 
        # in box
        Cos, Sin = torch.cos(gt_angles_per_image), torch.sin(gt_angles_per_image) # shape(num_gt, 1)
        Matric = torch.stack([Cos, -Sin, Sin, Cos], dim=-1).repeat(1, total_num_anchors, 1, 1).view(num_gt, total_num_anchors, 2, 2)
        offset = (grid_xy_per_image - gt_xy_per_image)[..., None] # shape(num_gt, n_anchor, 2, 1)
        offset = torch.matmul(Matric, offset).squeeze_(-1) # shape(n_gt, n_anchors, 2)

        b_lt = gt_wh_per_image / 2 + offset
        b_rb = gt_wh_per_image / 2 - offset
        bbox_deltas = torch.cat([b_lt, b_rb], dim=-1) # shape(n_gt, n_anchors, 4)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0 # shape(n_gt, n_anchors)
        is_in_boxes_all = is_in_boxes.sum(0) > 0 # shape(n_anchors)
        # in center
        center_radius = 2.5
        c_dist = center_radius * expanded_strides_per_image # shape(1, n_anchors_all, 1)
        c_lt = grid_xy_per_image - (gt_xy_per_image - c_dist)
        c_rb = (gt_xy_per_image + c_dist) - grid_xy_per_image

        center_deltas = torch.cat([c_lt, c_rb], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0 # shape(num_gts, n_anchors_all)
        is_in_centers_all = is_in_centers.sum(dim=0) > 0 # shape(n_anchors_all)

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all 
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center
    
    @staticmethod
    def postprocess(*args, **kwargs):
        return obbpostprocess(*args, **kwargs)


