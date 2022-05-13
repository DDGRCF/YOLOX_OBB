import torch
from yolox.models import Model
from yolox.utils import multiclass_obb_nms
def get_targets():
    labels = torch.randint(0, 80, (50, 1)).float()
    bboxes = torch.randint(0, 640, (50, 4)).float()
    target = torch.cat((labels, bboxes), dim=-1)[None]
    targets = torch.zeros((1, 120, 5))
    targets[:, :50, :] = target
    return targets



if __name__ == "__main__":
    modules_cfg = "/home/rcf/Desktop/YOLOX_OBB/configs/modules/yoloxs.yaml"
    losses_cfg = "/home/rcf/Desktop/YOLOX_OBB/configs/losses/yolox_losses.yaml"
    model = Model(modules_cfg, losses_cfg)
    model.train()
    device = torch.device("cuda:0")
    model = model.to(device)
    input = torch.rand(1, 3, 640, 640).to(device)
    targets = get_targets().to(device)
    input.requires_grad = True
    loss = model.get_losses(targets, input)
