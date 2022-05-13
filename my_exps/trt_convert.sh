#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolox_obb
cd ~/Desktop/YOLOX_OBB
cuda=$1
expn=YOLOX_outputs/trt_convert_resutls
exp_file=exps/example/yolox_obb/yolox_s_dota2_0.py
ckpt=YOLOX_outputs/yolox_s_dota2_0/latest_ckpt.pth

CUDA_VISIBLE_DEVICES=${cuda} python tools/trt.py -expn ${exp_file} -f ${exp_file} -c ${ckpt}




