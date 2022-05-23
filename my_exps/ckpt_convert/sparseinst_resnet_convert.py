import torch
import torch.nn as nn
from yolox.exp import get_exp, MaskExp
from yolox.models import Model


if __name__ == "__main__":
    exp_config = "/home/rcf/GPU_6/YOLOX_OBB/exps/example/yolox_inst/yolox_s_sparseinst_coco.py"
    ori_ckpt_path = "my_exps/ckpt_convert/origin_weights/yolov5l.pth"
    dst_ckpt_path = "my_exps/ckpt_convert/origin_weights/sparse_inst_darknet_base.pth"
    exp = get_exp(exp_config, "sparse_ckpt_convert")
    model = exp.get_model()
    ori_weights = torch.load(ori_ckpt_path)["model"]
    # print("----------------------------")
    # for k, v in ori_weights.items():
    #     print(k, v.shape)
    # print("----------------------------")
    for k, v in model.state_dict().items():
        print(k, v.shape)
    # for k, v in model.named_modules():
    #     if isinstance(v, nn.BatchNorm2d):
    #         print(k, v)
    # print("----------------------------")