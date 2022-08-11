#! /bin/bash
env=yolox_dect
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

expn=$1
exp=$2
ckpt=$3
image=$4
out_type=${5:-tensorrt}
batch_size=${6:-1} # now: only support one
device=${7:-cuda:0}
py_args=${@:8}

echo -e "\033[33mexp is ${exp}\033[0m"
echo -e "\033[33mexpn is ${expn}\033[0m"
echo -e "\033[33mckpt is ${ckpt}\033[0m"
echo -e "\033[33mout_type is ${out_type}\033[0m"
echo -e "\033[33mckpt is ${ckpt}\033[0m"
echo -e "\033[33mimage is ${image}\033[0m"
echo -e "\033[33mdevice is ${device}\033[0m"
echo -e "\033[33mbatch_size is ${batch_size}\033[0m"
echo -e "\033[33mother args: ${py_args}\033[0m"

python tools/export_deploy.py ${out_type} -expn ${expn} -b ${batch_size} -f ${exp} \
--inference-image ${image} -c ${ckpt} -d ${device} ${py_args}

