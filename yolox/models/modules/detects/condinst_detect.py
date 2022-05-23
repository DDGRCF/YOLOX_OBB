import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from .detectx import DetectX

class CondInstDetectX(DetectX):
    def __init__(
                 self, 
                 anchors=1, 
                 in_channels=(
                     128, 128, 128, 128, 128, 128),
                 reg_dim=5,
                 num_classes=15, 
                 num_cnt_params=169,
                 pre_nms_thre=0.45,   
                 pre_nms_topk=1000,
                 post_nms_topk=100,
                 **kwargs):
        super().__init__(
            anchors, 
            in_channels,
            reg_dim=reg_dim,
            num_classes=num_classes,
            **kwargs)
        self.num_cnt_params = num_cnt_params
        self.pre_nms_thre = pre_nms_thre
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

        reg_in_channels = in_channels[1::2]
        self.cnt_preds = nn.ModuleList()
        for reg_in_channel in reg_in_channels:
            cnt_pred = nn.Conv2d(
                in_channels=reg_in_channel,
                out_channels=self.n_anchors * self.num_cnt_params,
                kernel_size=1,
                stride=1,
            )
            self.cnt_preds.append(cnt_pred)
    
    def initialize_biases(self):
        super().initialize_biases()
        for module in self.cnt_preds:
            nn.init.normal_(module.weight, std=0.01)
            nn.init.constant_(module.bias, 0.)

    def _forward(self, xin):
        outputs = []
        cls_xs = xin[0::2]
        reg_xs = xin[1::2]
        for k, (cls_x, reg_x) in enumerate(zip(cls_xs, reg_xs)):
            cls_output = self.cls_preds[k](cls_x)
            reg_output = self.reg_preds[k](reg_x)
            obj_output = self.obj_preds[k](reg_x)
            cnt_output = self.cnt_preds[k](reg_x) # (n_p)
            output = torch.cat(
                [reg_output, 
                obj_output if self.training else obj_output.sigmoid(), 
                cls_output if self.training else cls_output.sigmoid(),
                cnt_output], 1)
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

    def get_losses(self, targets, inps):
        device = inps[0].device
        dtype = inps[0].dtype
        masks_t = targets[1]
        target_masks = [torch.from_numpy(m).to(device) for m in masks_t]
        # targets = [t for t in targets[0]]
        target_bboxes = targets[0]
        bbox_losses = self.get_bbox_losses(target_bboxes, inps)

    def get_bbox_losses(
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
        cls_preds = outputs[:, :, self.reg_dim + 1: self.reg_dim + 1 + self.num_classes]  # [batch, n_anchors_all, n_cls]
        cnt_preds = outputs[:, :, self.reg_dim + 1: self.reg_dim + 1 + self.num_classes: ]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]

        cls_targets = []
        obj_targets = []
        reg_targets = []
        reg_l1_targets = []
        fg_masks = []
        gt_inds = []
        img_inds = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
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
                gt_inds.append(matched_gt_inds + int(num_gts))
                img_inds.append(torch.full_like(matched_gt_inds, batch_idx))
                num_gts += num_gt
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
        if len(gt_inds):
            gt_inds = torch.cat(gt_inds, 0)
            img_inds = torch.cat(img_inds, 0)
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
            "loss_reg_extra": loss_reg_extra,
            "pos_inds": fg_masks.nonzero(as_tuple=False).squeeze(),
            "gt_inds": gt_inds,
            "img_inds": img_inds,
            "num_gts": int(num_gts)
        }

        return losses
