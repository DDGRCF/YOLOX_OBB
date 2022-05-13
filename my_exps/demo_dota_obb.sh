#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolox_obb
data_type=$1
path=$2 # /path/to/you
cuda=$3
echo -e "\033[31mcuda is ${cuda}\033[0m"
echo -e "\033[31mpath is ${path}\033[0m"
cd ..
echo -e "\033[31mcurrent path is ${PWD}\033[0m"
CUDA_VISIBLE_DEVICES=${cuda} python tools/demo_obb.py image -expn dota_demo_obb -c YOLOX_outputs/yolox_s_${data_type}/latest_ckpt.pth \
-f exps/example/yolox_obb/yolox_s_${data_type}.py --path ${path} --fuse --save_result --device gpu
cd -


