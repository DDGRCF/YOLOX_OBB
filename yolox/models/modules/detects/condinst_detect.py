import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from skimage import color
from loguru import logger

from yolox.ops.pytorch.nms.nms_wrapper import multiclass_nms
from .detectx import DetectX
from yolox.utils import (
    unfold_wo_center, aligned_bilinear, 
    get_image_color_similarity,
    cxcywh2xyxy)

class CondInstDetectX(DetectX):
    def __init__(
                 self, 
                 anchors=1, 
                 in_channels=(
                     128, 128, 128, 128, 128, 128, 8),
                 reg_dim=4,
                 num_classes=15, 
                 mask_num_layers=3,
                 mask_out_stride=4,
                 mask_feat_stride=8,
                 mask_feat_channel=8,
                 mask_out_channel=1,
                 size_of_interest=[64, 128, 256],
                 num_cnt_params=169,
                 enable_boxinst=False,
                 **kwargs):
        super().__init__(
            anchors, 
            in_channels[:-1],
            reg_dim=reg_dim,
            num_classes=num_classes,
            **kwargs)
        # bbox head
        self.num_cnt_params = num_cnt_params
        # dynamic mask
        self.enable_boxinst = enable_boxinst
        self.mask_in_channel = in_channels[-1]
        self.mask_feat_channel = mask_feat_channel
        self.mask_out_channel = mask_out_channel
        self.mask_num_layers = mask_num_layers

        self.mask_out_stride = mask_out_stride
        self.mask_feat_stride = mask_feat_stride
        self.register_buffer("size_of_interest", torch.Tensor(size_of_interest))
        self.weight_nums, self.bias_nums, self.num_gen_params\
            = self.generate_dynamic_filters() 
        if self.enable_boxinst:
            self.register_buffer("_iter", torch.zeros([1]))

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
        mask_feats = xin[-1]
        xin = xin[:-1]
        outputs = self._forward(xin)
        if self.training:
            outputs.append(mask_feats)
            return outputs
        else:
            grids = []
            strides = []
            levels = []
            device = outputs[0].device
            dtype = outputs[0].dtype
            for k,  output in enumerate(outputs):
                hsize, wsize = output.shape[-2:]
                yv, xv = torch.meshgrid((torch.arange(hsize, device=device, dtype=dtype), torch.arange(wsize, device=device, dtype=dtype)))
                grids.append(torch.stack((xv, yv), 2).view(hsize * wsize, 2))
                strides.append(torch.full((hsize * wsize, ), self.strides[k], device=device, dtype=dtype))
                levels.append(torch.full((hsize * wsize, ), k, device=device, dtype=dtype))
            self.hw = [out.shape[-2:] for out in outputs]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            outputs = self.decode_outputs(outputs, dtype=xin[0].type())
            batch_size = outputs.shape[0]
            grids = [torch.cat(grids) for _ in range(batch_size)]
            strides = [torch.cat(strides) for _ in range(batch_size)]
            levels = [torch.cat(levels) for _ in range(batch_size)]
            x0y0 = outputs[:, :, 0:2] - outputs[:, :, 2:4] / 2
            x1y1 = outputs[:, :, 0:2] + outputs[:, :, 2:4] / 2
            outputs = torch.cat((x0y0, x1y1, outputs[:, :, 4:]), dim=-1)
            outputs_ = [None for _ in range(len(outputs))]
            img_inds = [None for _ in range(len(outputs))]
            cnt_outs = [None for _ in range(len(outputs))]
            for i, output in enumerate(outputs):
                class_conf, class_pred = torch.max(output[:, self.reg_dim + 1: self.reg_dim + 1 +self.num_classes], 1, keepdim=True)
                bboxes = output[:, :self.reg_dim]
                obj_conf = output[:, [self.reg_dim]]
                cnt_out = output[:, self.reg_dim + 1 + self.num_classes :]
                img_ind = torch.full_like(class_conf, i)
                detections  = torch.cat((bboxes, obj_conf, class_conf, class_pred.float()), 1)
                # for conventient
                if len(bboxes) > self.bbox_pre_topk:
                    topk_keep_inds = torch.topk((obj_conf * class_conf).squeeze(-1), k=self.bbox_pre_topk, sorted=True)[1]
                else:
                    topk_keep_inds = torch.arange(len(bboxes), device=device)
                keep_inds = multiclass_nms(bboxes[topk_keep_inds], 
                                          class_conf.squeeze(-1)[topk_keep_inds], 
                                          class_pred.squeeze(-1)[topk_keep_inds], 
                                          score_factors=obj_conf.squeeze(-1)[topk_keep_inds], 
                                          iou_thr=self.bbox_iou_thre,
                                          max_num=self.bbox_nms_topk, 
                                          class_agnostic=False)
                keep_inds = topk_keep_inds[keep_inds]
                outputs_[i] = detections[keep_inds]
                img_inds[i] = img_ind[keep_inds]
                cnt_outs[i] = cnt_out[keep_inds]
                grids[i] = grids[i][keep_inds]
                strides[i] = strides[i][keep_inds]
                levels[i] = levels[i][keep_inds]
            outputs = torch.stack(outputs_, dim=0) # (bs, 100, 4 + 1 + 1 + 1)
            img_inds = torch.cat(img_inds, dim=0).type(torch.int64)
            levels = torch.cat(levels, dim=0).type(torch.int64)
            cnt_outs = torch.cat(cnt_outs, dim=0)
            grids = torch.cat(grids, dim=0)
            strides = torch.cat(strides, dim=0)
            grids_ps = grids * strides[..., None] + 0.5 * strides[..., None]
            if len(cnt_outs) > 0:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, cnt_outs, img_inds, grids_ps, levels, 
                    self.size_of_interest, self.mask_feat_stride, 1
                )
                mask_scores = mask_logits.sigmoid()
                mask_scores = [mask_scores[img_inds==i] for i in range(batch_size)]
                return mask_scores, outputs # (bs * n, 640, 640)
            else:
                return [cnt_outs.new_empty((0, 
                        mask_feats.shape[-2] * self.mask_feat_stride, 
                        mask_feats.shape[-1] * self.mask_feat_stride)) for _ in range(batch_size)], \
                    outputs.new_empty((batch_size, 0, self.reg_dim + 2))

    def get_losses(self, targets, inps):
        imgs = inps[-1]
        bbox_inps = inps[0][:-1]
        mask_feats = inps[0][-1]
        device = imgs.device
        dtype = imgs.dtype
        if self.enable_boxinst:
            target_bboxes = targets
            gt_masks=None
        else:
            gt_masks = targets[1]
            target_bboxes = targets[0]
        bbox_losses = self.get_bbox_losses(target_bboxes, bbox_inps)
        pos_inds = bbox_losses.pop("pos_inds")
        pos_levels = bbox_losses.pop("pos_levels")
        pos_gt_inds = bbox_losses.pop("pos_gt_inds")
        pos_img_inds = bbox_losses.pop("pos_img_inds")
        pos_num_gts = bbox_losses.pop("pos_num_gts")
        pos_grid_pos = bbox_losses.pop("pos_grid_pos")
        pos_cnt_preds = bbox_losses.pop("pos_cnt_preds")
        if len(pos_gt_inds) == 0:
            if self.enable_boxinst:
                losses = dict(loss_prj=mask_feats.sum() * 0., loss_pairwise=mask_feats.sum() * 0.)
            else:
                losses = dict(loss_mask=mask_feats.sum() * 0.)
        else:
            mask_logits = self.mask_heads_forward_with_coords(
                mask_feats, pos_cnt_preds, pos_img_inds, pos_grid_pos, pos_levels, 
                self.size_of_interest, self.mask_feat_stride, self.mask_out_stride, 
            )
            mask_scores = mask_logits.sigmoid()
            gt_masks, img_color_similarity = self.get_mask_targets(
                gt_masks, pos_gt_inds, self.mask_out_stride,
                imgs=imgs, gt_bboxes=target_bboxes[..., 1:5], 
                device=device, dtype=dtype
            )
            if self.enable_boxinst:
                self._iter += 1
                loss_prj_term = self.compute_project_term(mask_scores, gt_masks)
                pairwise_losses = self.compute_pairwise_term(mask_logits, self.pairwise_size, self.pairwise_dilation)
                color_similarity_mask = (img_color_similarity >= self.pairwise_color_thresh).type(dtype)
                color_similarity_weights = color_similarity_mask * gt_masks
                loss_pairwise = (pairwise_losses * color_similarity_weights).sum() / color_similarity_weights.sum().clamp(min=1.0)
                warmup_factor = min(self._iter.item() / float(self.pairwise_warmup), 1.0)
                loss_pairwise = self.loss_pairwise_weight * loss_pairwise *  warmup_factor
                losses = dict(
                    loss_prj=loss_prj_term,
                    loss_pairwise=loss_pairwise
                )
            else:
                loss_mask = self.mask_loss(mask_scores, gt_masks, avg_factor=len(pos_gt_inds))
                losses = dict(loss_mask=loss_mask)

        losses.update(bbox_losses)
        return losses

    def get_bbox_losses(
        self,
        labels,
        inputs):
        xy_shifts = []
        expanded_strides = []
        expanded_levels = []
        outputs = []
        origin_preds = []

        # output analyse
        batch_size = len(labels)
        for k, output in enumerate(inputs):
            if self.use_extra_loss:
                reg_output = output[:, :self.reg_dim, :, :].clone()
                hsize, wsize = reg_output.shape[-2:]
                # batch_size = reg_output.shape[0]
                reg_output = reg_output.view(batch_size, self.n_anchors, self.reg_dim, hsize, wsize)
                reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, self.reg_dim)
                origin_preds.append(reg_output)
            output, grid, expanded_stride = self.get_output_and_grid(
                output, k, self.strides[k], inputs[0].type())
            expanded_level = torch.full_like(grid[..., 0], k, dtype=torch.int64).view(1, -1)
            expanded_levels.append(expanded_level)
            xy_shifts.append(grid)
            expanded_strides.append(expanded_stride)
            outputs.append(output)

        if self.use_extra_loss:
            origin_preds = torch.cat(origin_preds, dim=1)
        outputs = torch.cat(outputs, dim=1)
        xy_shifts = torch.cat(xy_shifts, dim=1)
        expanded_strides = torch.cat(expanded_strides, dim=1)
        expanded_levels = torch.cat(expanded_levels, dim=1)

        bbox_preds = outputs[:, :, :self.reg_dim]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, [self.reg_dim]]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, self.reg_dim + 1: self.reg_dim + 1 + self.num_classes]  # [batch, n_anchors_all, n_cls]
        cnt_preds = outputs[:, :, self.reg_dim + 1 + self.num_classes: ]

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

        pos_inds = fg_masks.nonzero(as_tuple=False).squeeze(-1)
        pos_cnt_preds = cnt_preds.reshape(-1, self.num_cnt_params)[pos_inds]
        pos_levels = expanded_levels.expand(batch_size, -1).reshape(-1)[pos_inds]
        pos_grid_pos = (xy_shifts * expanded_strides[..., None] 
                        + expanded_strides[..., None] * 0.5).expand(batch_size, -1, -1).reshape(-1, 2)[pos_inds]
        losses = {
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_reg": loss_reg,
            "loss_reg_extra": loss_reg_extra,
            "pos_cnt_preds":pos_cnt_preds, 
            "pos_grid_pos":pos_grid_pos,
            "pos_inds": pos_inds,
            "pos_levels":pos_levels,
            "pos_gt_inds": gt_inds,
            "pos_img_inds": img_inds,
            "pos_num_gts": int(num_gts)
        }

        return losses

    def get_mask_targets(self, gt_masks, gt_inds, out_stride, imgs=None, gt_bboxes=None, device=None, dtype=None):

        H, W = imgs.shape[2:]
        if self.enable_boxinst:
            batches = imgs.shape[0]
            imgs_masks =  imgs.new_ones((batches, H, W)) # (bs, H, W)
            # imgs = imgs[:, ::-1, :, :] # bgr to rbg (bs, 3, H, W)
            imgs = imgs.flip(1)
            gt_masks, similarities = self.get_gt_masks_from_bboxes(gt_bboxes, imgs, imgs_masks, out_stride)
            similarities  = torch.cat(similarities, dim=0).to(device).to(dtype)
            gt_masks = torch.cat(gt_masks, dim=0).to(device).type(dtype)
            similarities = similarities[gt_inds]
            gt_masks = gt_masks[gt_inds][:, None]
        else:
            gt_masks = np.concatenate(gt_masks, 0)
            gt_masks = torch.from_numpy(gt_masks).to(device).type(dtype)[:, None]
            gt_masks = nn.functional.interpolate(
                            gt_masks, 
                            size=(H // out_stride, W // out_stride))
            # start = int(out_stride // 2)
            # gt_masks = gt_masks[:, start::out_stride, start::out_stride]
            gt_masks = gt_masks[gt_inds]
            similarities = None
        
        return gt_masks, similarities

    def mask_heads_forward(self, features, weights, biases):
        """
        Args:
            features ([type]): [description]: (num_proposals, in_channel, H * W)
            weights ([type]): [description]: (num_proposals, in_channel, out_channel)
            biases ([type]): [description]: (num_proposals, out_channel)
        """
        features = features.transpose(1, 2)
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = torch.bmm(x, w) # (num_proposals, H*W, out_channel)
            x += b
            if i < n_layers - 1:
                x = F.relu(x)
        return x.transpose(1, 2) # (num_prooposals, 1, H * W)


    def mask_heads_forward_with_coords(
        self, mask_feats, mask_head_params, img_inds, grid_positions, 
        fpn_levels, size_of_interest, mask_feat_stride, mask_stride_out
    ):
        hsize, wsize = mask_feats.shape[2:]        
        yv, xv = torch.meshgrid((torch.arange(hsize, device=mask_feats.device),
                         torch.arange(wsize, device=mask_feats.device)))
        xv, yv = xv.reshape(-1), yv.reshape(-1)
        grid = torch.stack((xv, yv), dim=1) * mask_feat_stride + 0.5 * mask_feat_stride
        global_grid = grid.type(mask_feats.dtype)

        hsize, wsize = mask_feats.shape[2:]
        relative_coords = grid_positions.view(-1, 1, 2) - global_grid.view(1, -1, 2)
        # relative_coords = relative_coords.permute(0, 2, 1)
        relative_coords = relative_coords.transpose(1, 2)
        soi = (size_of_interest.type(mask_feats.dtype))[fpn_levels]
        relative_coords = relative_coords / soi.view(-1, 1, 1)
        relative_coords = relative_coords.type(mask_feats.dtype)
        mask_head_inputs = torch.cat([
            relative_coords, mask_feats[img_inds].reshape(-1, self.mask_in_channel, hsize * wsize)
        ], dim=1)                # weight_splits[l] = weight_splits[l].reshape(-1, out_channel, out_channel)

        weights, biases = self.parse_dynamic_params(
            mask_head_params, self.weight_nums, self.bias_nums, 
            in_channel=self.mask_feat_channel + 2,
            feat_channel = self.mask_feat_channel, out_channel=self.mask_out_channel
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases)
        mask_logits = mask_logits.reshape(-1, self.mask_out_channel, hsize, wsize)
        mask_logits = aligned_bilinear(mask_logits, mask_feat_stride // mask_stride_out)

        return  mask_logits

    def get_gt_masks_from_bboxes(self, bboxes, imgs, masks, stride):
        H, W = imgs.shape[2:]
        device, dtype = masks.device, masks.dtype
        start = int(stride // 2)
        assert imgs.size(2) % stride == 0
        assert imgs.size(3) % stride == 0
        # ignore imgs.float()
        d_imgs = F.avg_pool2d(imgs, kernel_size=stride, stride=stride, padding=0)
        d_masks = masks[:, start::stride, start::stride]

        similarities = []
        bitmasks = []
        bboxes = cxcywh2xyxy(bboxes) # [center_x, center_y, w, h] to [x, y, x, y]
        for i, per_img_bboxes in enumerate(bboxes):
            img_lab = color.rgb2lab(d_imgs[i].byte().permute(1, 2, 0).cpu().numpy())
            img_lab = torch.as_tensor(img_lab, device=device, dtype=dtype)
            img_lab = img_lab.permute(2, 0, 1)[None]
            img_color_similarity = get_image_color_similarity(
                img_lab, d_masks[i], 
                self.pairwise_size, self.pairwise_dilation
            )

            nlabels = int((per_img_bboxes.sum(1) > 0).sum(0))
            if nlabels == 0:
                continue
            per_img_bboxes = per_img_bboxes[:nlabels]
            per_im_bitmasks = []
            for per_bbox in per_img_bboxes:
                bitmask_full = torch.zeros((H, W), device=device, dtype=torch.float)
                bitmask_full[int(per_bbox[1]): int(per_bbox[3]) + 1, int(per_bbox[0]): int(per_bbox[2]) + 1] = 1.0
                bitmask = bitmask_full[start::stride, start::stride]
                assert bitmask.size(0) * stride == H 
                assert bitmask.size(1) * stride == W
                per_im_bitmasks.append(bitmask)
            
            bitmasks.append(torch.stack(per_im_bitmasks, dim=0))
            similarities.append(torch.cat([img_color_similarity for _ in range(nlabels)], dim=0))
        
        return bitmasks, similarities

    def generate_dynamic_filters(self):
        weight_nums, bias_nums = [], []
        for l in range(self.mask_num_layers):
            if l == 0:
                weight_nums.append((self.mask_in_channel + 2) * self.mask_feat_channel) 
                bias_nums.append(self.mask_feat_channel)
            elif l == self.mask_num_layers - 1:
                weight_nums.append(self.mask_feat_channel * self.mask_out_channel)
                bias_nums.append(self.mask_out_channel)
            else:
                weight_nums.append(self.mask_feat_channel * self.mask_feat_channel)
                bias_nums.append(self.mask_feat_channel)
        num_gen_params = sum(weight_nums) + sum(bias_nums)
        assert len(weight_nums) == len(bias_nums)
        return weight_nums, bias_nums, num_gen_params 

    def parse_dynamic_params(self, 
                             params, 
                             weight_nums, 
                             bias_nums,
                             in_channel, 
                             feat_channel,
                             out_channel=1):

        num_layers = len(weight_nums)
        params_splits = list(
            torch.split_with_sizes(params, weight_nums + bias_nums, dim=1)
        )
        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l == 0:
                weight_splits[l] = weight_splits[l].reshape(-1, in_channel, feat_channel)
                bias_splits[l] = bias_splits[l].reshape(-1, 1, feat_channel)
            elif l == num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(-1, feat_channel, out_channel)
                bias_splits[l] = bias_splits[l].reshape(-1, 1, out_channel)
            else:
                weight_splits[l] = weight_splits[l].reshape(-1, feat_channel, feat_channel)
                bias_splits[l] = bias_splits[l].reshape(-1, 1, feat_channel)

        return weight_splits, bias_splits 

    def compute_project_term(self, mask_scores, gt_bitmasks):
        num_gts = mask_scores.size(0)
        mask_losses_y = self.prj_loss(
            mask_scores.max(dim=2, keepdim=True)[0],
            gt_bitmasks.max(dim=2, keepdim=True)[0],
            num_gts
        )
        mask_losses_x = self.prj_loss(
            mask_scores.max(dim=3, keepdim=True)[0],
            gt_bitmasks.max(dim=3, keepdim=True)[0],
            num_gts
        )
        return mask_losses_x + mask_losses_y

    def compute_pairwise_term(self, mask_logits, pairwise_size, pairwise_dilation):
        assert mask_logits.dim() == 4

        log_fg_prob = F.logsigmoid(mask_logits)
        log_bg_prob = F.logsigmoid(-mask_logits)

        log_fg_prob_unfold = unfold_wo_center(
            log_fg_prob, kernel_size=pairwise_size,
            dilation=pairwise_dilation
        ) # (bs, 1, 8, h, w)
        log_bg_prob_unfold = unfold_wo_center(
            log_bg_prob, kernel_size=pairwise_size,
            dilation=pairwise_dilation
        ) # (bs, 1, 8, h, w)

        # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
        # we compute the the probability in log space to avoid numerical instability
        log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
        log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

        # this equation is equal to log(p_i * p_j + (1 - p_i) * (1 - p_j))
        # max is used to prevent overflow
        max_ = torch.max(log_same_fg_prob, log_same_bg_prob)  #
        log_same_prob = torch.log(
            torch.exp(log_same_fg_prob - max_) +
            torch.exp(log_same_bg_prob - max_)
        ) + max_ 

        return -log_same_prob[:, 0]    

    
    @staticmethod
    def postprocess(inputs, num_classes=80, conf_thre=0.1, 
                    mask_thre=0.50, eps=1e-6, **kwargs):
        outputs = [(None, None) for _ in range(len(inputs[0]))]
        for i, (bs_masks, bs_bboxes) in enumerate(zip(inputs[0], inputs[1])):
            if bs_masks.size(0) == 0:
                continue
            bs_scores = bs_bboxes[:, 4] * bs_bboxes[:, 5]
            bs_labels = bs_bboxes[:, -1]
            bs_bboxes = bs_bboxes[:, :4]
            keep = bs_scores > conf_thre
            bs_scores = bs_scores[keep]
            bs_labels = bs_labels[keep]
            bs_masks = bs_masks[keep]
            bs_bboxes = bs_bboxes[keep]
            if bs_scores.size(0) == 0:
                continue
            bs_masks = (bs_masks > mask_thre).type(bs_scores.dtype)
            outputs[i] = (bs_masks, torch.cat((bs_bboxes, bs_scores[..., None], bs_labels[..., None]), dim=-1))
        return outputs