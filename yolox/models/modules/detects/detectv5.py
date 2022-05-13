import torch
import torch.nn as nn
import math
from .detect import Detect


class DetectV5(Detect):
    def __init__(self, anchors=(), in_channels=(), inplace=True, reg_dim=4, num_classes=80, **kwargs):
        super().__init__(**kwargs)
        self.reg_dim = reg_dim
        self.num_classes = num_classes
        self.out_channels =  reg_dim + 1 + num_classes
        self.num_layers = len(anchors)
        self.num_anchors = len(anchors[0]) // 2
        self.loss_balance = [4, 1, 4]
        self.grids = [torch.zeros(1)] * len(anchors)
        self.expanded_strides = [torch.zeros(1)] * len(anchors)
        self.hw = [[]] * len(anchors)
        anchors_ = torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        self.register_buffer("anchors", anchors_)
        self.register_buffer("anchors_grids", anchors_.clone().view(self.num_layers, 1, -1, 1, 1, 2))
        self.preds = nn.ModuleList(nn.Conv2d(x, self.num_anchors * self.out_channels, 1) for x in in_channels)
        self.inplace = inplace

    def _forward(self, xin):
        outputs = []
        for k, input in enumerate(xin):
            input = self.preds[k](input)
            hsize, wsize = input.shape[-2:]
            self.hw[k] = [hsize, wsize]
            output = input.sigmoid() # (bs, na * nc, h, w)
            # output = input.view(bs, self.num_anchors, 
            #     self.out_channels, hsize, wsize).permute(0, 1, 3, 4, 2) # (bs, na, h, w, ch)
            # if self.training:
            #     # output = output.reshape(bs, -1, self.out_channels
            #     output, grid, expanded_stride = self.get_output_and_grid(output, i, stride, output.type()) 
            outputs.append(output)
        return outputs
    
    def forward(self, xin):
        outputs = self._forward(xin)
        if self.training:
            return outputs
        else:
            # for output in outputs:
            #     output = output.view()
            # outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1) # (bs, -1, ch)
            return self.decode_outputs(outputs)

    # def get_output_and_grid(self, output, k, stride, dtype):
    #     grid = self.grids[k]
    #     expanded_stride = self.expanded_strides[k]
    #     assert(output.shape[1] % self.num_anchors == 0)
    #     hsize, wsize = output.shape[-2:]
    #     if (grid.shape[2: 4] != output.shape[2: 4]):
    #         grid = Detect._make_grid(wsize, hsize, dtype=dtype) # (bs, )
    #         expanded_stride = torch.full_like(grid[..., 0], stride).type(dtype)
    #         self.grids[k] = grid # (bs, na, h, w, 2)
    #         self.expanded_strides[k] = expanded_stride # (bs, na, h, w, 2)
        
    #     # y = output.sigmoid() # yolov5 sigmoid all     
    #     # grid = grid.view(1, -1, 2)
    #     # # expanded_stride = expanded_stride.view(1, -1)
    #     # y[..., 0: 2] = (y[..., 0: 2] * 2. - 0.5 + self.grids[k]) * stride 
    #     # y[..., 2: 4] = (y[..., 2: 4] * 2) ** 2 * self.anchor_grids[k]
    #     return output
    def initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(self.preds, self.strides):  # from
            b = mi.bias.view(self.num_anchors, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def decode_outputs(self, outputs_):
        grids = []
        strides = []
        anchors_grids = []
        bs = outputs_[0].shape[0]
        dtype = outputs_[0].dtype
        device = outputs_[0].device
        outputs = []
        for i, ((hsize, wsize), stride, output) in enumerate(zip(self.hw, self.strides, outputs_)):
            output = output.view(bs, self.num_anchors, self.out_channels, hsize, wsize).permute(0, 1, 3, 4, 2)
            output = output.reshape(bs, -1, self.out_channels)
            outputs.append(output)
            grid = Detect._make_grid(wsize, hsize, (1, -1, 2), dtype)
            strides.append(torch.full((*(grid.shape[:2]), 1), stride).repeat(1, self.num_anchors, 1))
            grid = grid.repeat(1, self.num_anchors, 1)
            grids.append(grid)
            anchors = self.anchors_grids[i]
            anchors = anchors.expand(bs, -1, hsize, wsize, -1).reshape(bs, self.num_anchors * hsize * wsize, 2)
            anchors_grids.append(anchors)
        grids = torch.cat(grids, dim=1).type(dtype).to(device)
        strides = torch.cat(strides, dim=1).type(dtype).to(device)
        anchors_grids = torch.cat(anchors_grids, dim=1).type(dtype) # (bs, na * h * w, w)
        outputs = torch.cat(outputs, dim=1)
        outputs[..., :2] = (outputs[..., :2] * 2 - 0.5 + grids) * strides 
        outputs[..., 2:4] = (outputs[..., 2:4] * 2) ** 2 * anchors_grids
        return outputs


    def get_losses(self, labels, preds):
        device = preds[0].device # preds list(3) (bs, na * nc, h, w)
        dtype = preds[0].dtype
        strides = self.strides.to(device).type(dtype)
        anchors = self.anchors / strides.view(-1, 1, 1) # (3, 3, 2) / (3, 1, 1)
        # batch_preds_list = []

        # for b in range(batch_size):
        #     level_preds_list = []
        #     for l in range(self.num_layers):
        #         pred = preds[l][b]
        #         pred = pred.view(self.num_anchors, self.out_channels, *self.hw[l])
        #         pred = pred.permute(0, 2, 3, 1)
        #         level_preds_list.append(pred)
        #     level_preds_list = torch.stack(level_preds_list, 0) # (bs, na, h, w, nc)
        #     batch_preds_list.append(level_preds_list)

        loss_reg = 0.0
        loss_cls = 0.0
        loss_obj = 0.0

        for bs_ind, label in enumerate(labels):
            num_anchors, num_targets = self.num_anchors, int((label.sum(1) > 0.).sum(0)) 
            label = label[:num_targets] # Get the turth label
            # batch_size_inds = torch.full_like(label[:, [0]], bs_ind)
            # label = torch.cat((batch_size_inds, label), dim=1) # (nt, 6)

            # norm_gain = outputs.new_ones(7) # (7)
            anchor_id = torch.arange(num_anchors, device=device).type(dtype) # (na, )
            anchor_id = anchor_id.view(num_anchors, 1).repeat(1, num_targets) # (na, nt)
            label = torch.cat((label.repeat(self.num_anchors, 1, 1), anchor_id[..., None]), dim=-1)  # (na, nt, 6) [l, x, y, w, h, a]

            center_bias = 0.5
            point_offset = torch.tensor([[0, 0],
                                [1, 0], [0, 1], [-1, 0], [0, -1], 
                                ], device=device).type(dtype) * center_bias  # offsets (5, 2)

            for i in range(self.num_layers):
                anchor = anchors[i] # (2, )
                stride = strides[i] # (s, )
                hsize, wsize = self.hw[i]
                pred = preds[i][bs_ind].view(self.num_anchors, self.out_channels, hsize, wsize).permute(0, 2, 3, 1).contiguous() # (na, h, w, oc)
                # norm_gain[2:6] = torch.tensor(output[i].shape)[[3, 2, 3, 2]]  # xyxy gain
                # norm_gain[2:6] = torch.tensor([hsize, wsize, hsize, wsize]).type(dtype)
                # norm_gain[2:6] = torch.tensor([1. / stride] * 4).type(dtype)
                # Match targets to anchors
                target = label.clone()
                target[..., 1:5] = target[..., 1:5] / stride 
                if num_targets:
                    # Matches
                    hw_ratio = target[:, :, 3:5] / anchor[:, None]  # wh ratio (na, nt, 2) / (na, 1, 2)
                    hw_mask = torch.max(hw_ratio, 1 / hw_ratio).max(2)[0] < self.anchor_thre # compare
                    # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                    target = target[hw_mask]  # filter # (va, 7)
                    grid_xy = target[:, 1:3]  # grid xy # (va, 2)
                    grid_xy_inverse = torch.empty_like(grid_xy) # (va, 2)
                    grid_xy_inverse[:, 0] = wsize - grid_xy[:, 0]
                    grid_xy_inverse[:, 1] = hsize - grid_xy[:, 1]

                    # j, k = ((grid_xy % 1 < center_bias) & (grid_xy > 1)).T 
                    # l, m = ((grid_xy_inverse % 1 < center_bias) & (grid_xy_inverse > 1)).T
                    grid_left_offset, grid_top_offset = ((grid_xy % 1 < center_bias) & (grid_xy > 1)).T # (va, ) (va, )
                    grid_right_offset, grid_bottom_offset = ((grid_xy_inverse % 1 < center_bias) & (grid_xy_inverse > 1)).T # (va, ) (va, )
                    point_offset_mask = torch.stack((torch.ones_like(grid_left_offset), grid_left_offset, grid_top_offset, grid_right_offset, grid_bottom_offset)) # (5, va)
                    # j = torch.stack((torch.ones_like(j), j, k, l, m))
                    target = target.repeat(5, 1, 1)[point_offset_mask] # (vf, 7)
                    # offsets = (torch.zeros_like(gxy)[None] + point_bias_[:, None])[j]
                    point_offset_expand = (torch.zeros_like(grid_xy)[None] + point_offset[:, None])[point_offset_mask] # (1, va, 2) + (5, 1, 2) -> (vf, 2)
                else:
                    target = target[0]
                    point_offset_expand = 0

                # Define
                pos_class_inds = target[:, 0].long() # (vf, )
                # b, c = t[:, :2].long().T  # image, class
                pos_grid_xy = target[:, 1:3] # (vf, 2)
                pos_grid_wh = target[:, 3:5] # (vf, 2)
                pos_grid_xy_expand = (pos_grid_xy - point_offset_expand).long() # (vf, 2)
                # gxy = t[:, 2:4]  # grid xy
                # gwh = t[:, 4:6]  # grid wh
                # gij = (gxy - offsets).long()
                pos_grid_x_expand, pos_grid_y_expand = pos_grid_xy_expand.T # (vf, ) (vf, )
                pos_grid_x_expand = pos_grid_x_expand.clamp_(0, wsize - 1)
                pos_grid_y_expand = pos_grid_y_expand.clamp_(0, hsize - 1)
                # gi, gj = gij.T  # grid xy indices
                # Append
                pos_anchor_inds = target[:, 5].long()  # anchor indices # (vf)
                pos_anchor = anchor[pos_anchor_inds] # (vf, 2)
                pos_bboxes_target = torch.cat((pos_grid_xy - pos_grid_xy_expand, pos_grid_wh), dim=-1) # (vf, 4)
                # pos_indices.append((pos_bs_inds, pos_anchor_inds, pos_grid_x_expand.clamp_(0, wsize - 1), pos_grid_y_expand.clamp_(0, hsize - 1)))  # image, anchor, grid indices
                # target_boxes.append(torch.cat((pos_grid_xy - pos_grid_xy_expand, pos_grid_wh), ))
                # pos_anchors.append(anchors[pos_anchor_inds])
                # target_cls.append(pos_class_inds)
                # tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
                # anch.append(anchors[a])  # anchors
                # tcls.append(c)  # class
                target_obj = torch.zeros_like(pred[..., 0])  # (na, h, w)
                if len(pos_anchor_inds):
                    pos_pred = pred[pos_anchor_inds, pos_grid_y_expand, pos_grid_x_expand]
                    pos_pred_xy = pos_pred[:, :2] * 2 - 0.5
                    pos_pred_wh = (pos_pred[:, 2:4] * 2) ** 2 * pos_anchor
                    pos_pred_bbox = torch.cat((pos_pred_xy, pos_pred_wh), dim=-1)
                    # pos_bbox_iou = matrix_bboxes_iou(pos_pred_bbox, pos_bboxes_target, xyxy=False)
                    # loss_reg += (1 - pos_bbox_iou).mean()
                    loss_reg_, pos_bbox_iou = self.reg_loss(pos_pred_bbox, pos_bboxes_target)
                    loss_reg += loss_reg_
                    pos_iou_score = pos_bbox_iou.detach().clamp(0.).type(dtype)
                    target_obj[pos_anchor_inds, pos_grid_y_expand, pos_grid_x_expand] = pos_iou_score
                    cls_on_hot_target = torch.full_like(pos_pred[..., 5:], 0)
                    cls_on_hot_target[range(pos_pred.shape[0]), pos_class_inds] = 1.0
                    loss_cls += self.cls_loss(pos_pred[..., 5:], cls_on_hot_target)
                else:
                    loss_cls += pred[..., 5:].mean() * 0.
                    loss_reg += pred[..., :4].mean() * 0.

                loss_obj += self.obj_loss(pred[..., 4], target_obj) * self.loss_balance[i]

        # bs = target_obj.shape[0]
        bs = 1
        loss = dict (
            loss_obj = loss_obj * bs,
            loss_cls = loss_cls * bs,
            loss_reg = loss_reg * bs,
        ) 
        return loss 