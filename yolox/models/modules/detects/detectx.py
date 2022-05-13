import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .detect import Detect
from loguru import logger
from yolox.utils import bboxes_iou
from functools import partial

class DetectX(Detect):
    def __init__(self, 
                 anchors=1, 
                 in_channels=(
                     128, 128, 128, 128, 128, 128),
                 inplace=True,
                 reg_dim=4,
                 num_classes=80, 
                 **kwargs):
        self.use_extra_loss = False
        self.reg_dim = reg_dim
        super().__init__(**kwargs)
        if isinstance(anchors, (list, tuple)):
            self.n_anchors = len(anchors)
        elif isinstance(anchors, (float, int)):
            self.n_anchors = anchors
        self.num_classes = num_classes
        self.inplace = inplace
        # parameters
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.expanded_strides = [torch.zeros(1)] * len(in_channels)
        # modules
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        cls_in_channels = in_channels[0::2]
        reg_in_channels = in_channels[1::2]
        for cls_in_channel, reg_in_channel in zip(cls_in_channels, reg_in_channels):
            cls_pred = nn.Conv2d(
                in_channels=cls_in_channel,
                out_channels=self.n_anchors * self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0)
            reg_pred = nn.Conv2d(
                in_channels=reg_in_channel,
                out_channels=self.n_anchors * self.reg_dim,
                kernel_size=1,
                stride=1,
                padding=0)
            obj_pred = nn.Conv2d(
                in_channels=reg_in_channel,
                out_channels=self.n_anchors * 1,
                kernel_size=1,
                stride=1,
                padding=0)
            self.cls_preds.append(cls_pred)
            self.reg_preds.append(reg_pred)
            self.obj_preds.append(obj_pred)
        self.cal_overlaps = partial(bboxes_iou, xyxy=False)
    
    def initialize_biases(self):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2) # shape(1, w * h, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride)) # shape(1, w * h, 1)

        grids = torch.cat(grids, dim=1).type(dtype) # shape(1, sigma(w*h), 2)
        strides = torch.cat(strides, dim=1).type(dtype) # same as above

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
    
    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        expanded_stride = self.expanded_strides[k]
        batch_size = output.shape[0]
        assert(output.shape[1] % self.n_anchors == 0)
        num_channels = output.shape[1] // self.n_anchors
        hsize, wsize = output.shape[-2:]
        if grid.shape[2: 4] != output.shape[2: 4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            expanded_stride = torch.full_like(grid[..., 0], stride)
            self.grids[k] = grid
            self.expanded_strides[k] = expanded_stride
        output = output.view(batch_size, self.n_anchors, num_channels, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        # Extra Val  
        grid = grid.view(1, -1, 2)
        expanded_stride = expanded_stride.view(1, -1)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2: 4] = torch.exp(output[..., 2: 4]) * stride
        return output, grid, expanded_stride

    def get_reg_l1_target(self, l1_target, gt, stride, xy_shifts, eps=1e-8):
        l1_target[:, :2] = gt[:, :2] / stride[..., None] - xy_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride.squeeze(-1) + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride.squeeze(-1) + eps)
        return l1_target

    def _forward(self, xin):
        outputs = []
        cls_xs = xin[0::2]
        reg_xs = xin[1::2]
        for k, (cls_x, reg_x) in enumerate(zip(cls_xs, reg_xs)):
            cls_output = self.cls_preds[k](cls_x)
            reg_output = self.reg_preds[k](reg_x)
            obj_output = self.obj_preds[k](reg_x)
            output = torch.cat(
                [reg_output, 
                obj_output if self.training else obj_output.sigmoid(), 
                cls_output if self.training else cls_output.sigmoid()], 1)
            outputs.append(output)
        return outputs
    

    def forward(self, xin):
        outputs = self._forward(xin)
        if self.training:
            return outputs
        else:
            self.hw = [out.shape[-2:] for out in outputs]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            return self.decode_outputs(outputs, dtype=xin[0].type())

    def get_losses(
        self,
        labels,
        inputs):
        xy_shifts = []
        expanded_strides = []
        outputs = []
        origin_preds = []

        # output analyse
        for k, output in enumerate(inputs):
            if self.use_extra_loss:
                reg_output = output[:, :self.reg_dim, :, :].clone()
                hsize, wsize = reg_output.shape[-2:]
                batch_size = reg_output.shape[0]
                reg_output = reg_output.view(batch_size, self.n_anchors, self.reg_dim, hsize, wsize)
                reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, self.reg_dim)
                origin_preds.append(reg_output)
            output, grid, expanded_stride = self.get_output_and_grid(
                output, k, self.strides[k], inputs[0].type())
            xy_shifts.append(grid)
            expanded_strides.append(expanded_stride)
            outputs.append(output)

        if self.use_extra_loss:
            origin_preds = torch.cat(origin_preds, dim=1)
        outputs = torch.cat(outputs, dim=1)
        xy_shifts = torch.cat(xy_shifts, dim=1)
        expanded_strides = torch.cat(expanded_strides, dim=1)

        bbox_preds = outputs[:, :, :self.reg_dim]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, [self.reg_dim]]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, self.reg_dim + 1:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]

        cls_targets = []
        obj_targets = []
        reg_targets = []
        reg_l1_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, self.reg_dim))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                if self.use_extra_loss:
                    reg_l1_target = outputs.new_zeros((0, self.reg_dim))
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:1+self.reg_dim]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        xy_shifts,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        xy_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1) # shape(num_pos, num_classes)
                obj_target = fg_mask.unsqueeze(-1) # shape(n_anchors_all, 1)
                reg_target = gt_bboxes_per_image[matched_gt_inds] # shape(num_pos, 4)
                if self.use_extra_loss:
                    reg_l1_target = self.get_reg_l1_target(
                        outputs.new_zeros((num_fg_img, self.reg_dim)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        xy_shifts=xy_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)
            if self.use_extra_loss:
                reg_l1_targets.append(reg_l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_extra_loss:
            reg_l1_targets = torch.cat(reg_l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_reg = self.reg_loss(bbox_preds.view(-1, self.reg_dim)[fg_masks], reg_targets, num_fg)
        loss_obj = self.obj_loss(obj_preds.view(-1, 1), obj_targets, num_fg)
        loss_cls = self.cls_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets, num_fg)
        loss_reg_extra = self.reg_loss_extra(origin_preds.view(-1, self.reg_dim)[fg_masks], reg_l1_targets, num_fg) if self.use_extra_loss else 0.0

        losses = {
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "loss_reg_extra": loss_reg_extra
        }

        return losses

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        xy_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            xy_shifts = xy_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            xy_shifts,
        )
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask] # shape(num_in, 4)
        cls_preds_ = cls_preds[batch_idx][fg_mask] # shape(num_in, num_classes)
        obj_preds_ = obj_preds[batch_idx][fg_mask] # shape(num_in, 1)
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        pair_wise_ious = self.cal_overlaps(gt_bboxes_per_image, bboxes_preds_per_image)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        ) # shape(num_gt, num_in)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            ) # shape(num_gt, num_in, num_classes)
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1) # shape(num_gt, num_in)
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
        gt_bboxes_per_image,
        expanded_strides,
        xy_shifts,
    ):
        gt_xy_per_image = gt_bboxes_per_image[:, None, 0:2]
        gt_wh_per_image = gt_bboxes_per_image[:, None, 2:4]
        expanded_strides_per_image = expanded_strides[..., None]
        xy_shifts_per_image = xy_shifts * expanded_strides_per_image # (#all_anchors, 2)
        grid_xy_per_image = xy_shifts_per_image + 0.5 * expanded_strides_per_image

        b_off = grid_xy_per_image - gt_xy_per_image
        b_lt = gt_wh_per_image / 2 + b_off
        b_rb = gt_wh_per_image / 2 - b_off
        bbox_deltas = torch.cat([b_lt, b_rb], 2)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(0) > 0

        center_radius = 2.5
        c_dist = center_radius * expanded_strides_per_image
        c_lt = grid_xy_per_image - (gt_xy_per_image - c_dist)
        c_rb = (gt_xy_per_image + c_dist) - grid_xy_per_image
        center_deltas = torch.cat([c_lt, c_rb], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
