#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolox_obb
expn=$1
exp=$2
ckpt=$3
cuda=$4
phases=(test eval)
CUDA_VISIBLE_DEVICES=${cuda} python tools/eval_obb.py -expn ${expn} \
-b 1 -d 1 -f ${exp} -c ${ckpt} --fuse --test --conf 0.05 --nms 0.1 \
 --options is_merge=True is_submiss=True
