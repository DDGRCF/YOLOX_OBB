#! /bin/bash
env=yolox_dect
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

expn=$1
exp=$2
ckpt=$3
cuda_device=${4:-0}
num_device=${5:-1}
batch_size=${6:-8}
py_args=${@:7}

echo -e "\033[33mexp is ${exp}\033[0m"
echo -e "\033[33mexpn is ${expn}\033[0m"
echo -e "\033[33mckpt is ${ckpt}\033[0m"
echo -e "\033[33mcuda_device is cuda: ${cuda_device}\033[0m"
echo -e "\033[33mnum_device is ${num_device}\033[0m"
echo -e "\033[33mbatch_size is ${batch_size}\033[0m"
echo -e "\033[33mother args: ${py_args}\033[0m"

CUDA_VISIBLE_DEVICES=${cuda_device} python tools/eval.py -expn ${expn} \
-b ${num_device} -d ${batch_size} -f ${exp} -c ${ckpt} ${py_args}
