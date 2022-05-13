#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolox_obb
data_type=$1
phase=$2
cuda=$3
phases=(test eval)
if [ $phase != null ]; then
    if [[ "${phases[@]}" =~ "${phase}" ]]; then
        echo -e "\033[31mBegin ${phase}.......\033[0m"
        sleep 2s
    elif [[ ! "${phases[@]}" =~ "${phase}" ]]; then
        echo -e "\033[32m${phase} is unsupported\033[0m"
        exit
    fi
fi
cd ..
if [ $phase = "test" ]; then
    CUDA_VISIBLE_DEVICES=${cuda} python tools/eval_obb.py -expn dota_eval_obb \
    -b 1 -d 1 -f exps/example/yolox_obb/yolox_s_${data_type}.py \
    -c YOLOX_outputs/yolox_s_${data_type}/latest_ckpt.pth \
    --fuse --test --is_merge --is_submiss --conf 0.05 --nms 0.1
elif [ $phase = "eval" ]; then
    CUDA_VISIBLE_DEVICES=${cuda} python tools/eval_obb.py -expn dota_eval_obb \
    -b 1 -d 1 -f exps/example/yolox_obb/yolox_s_${data_type}.py \
    -c YOLOX_outputs/yolox_s_${data_type}/latest_ckpt.pth \
    --fuse --is_merge --conf 0.05 --nms 0.1
fi
