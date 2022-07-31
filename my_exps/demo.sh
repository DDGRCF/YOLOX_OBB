#! /bin/bash
env=yolox_dect
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

expn=$1
exp=$2
ckpt=$3
cuda_device=${4:-0}
path=$5
py_args=${@:6}

echo -e "\033[33mexp is ${exp}\033[0m"
echo -e "\033[33mexpn is ${expn}\033[0m"
echo -e "\033[33mckpt is ${ckpt}\033[0m"
echo -e "\033[33mcuda_device is cuda: ${cuda_device}\033[0m"
echo -e "\033[33mother args: ${py_args}\033[0m"

CUDA_VISIBLE_DEVICES=${cuda_device} python tools/demo.py image -expn ${expn} -c ${ckpt} \
-f ${exp} --path ${path} --fuse  --device cuda ${py_args}
cd -


