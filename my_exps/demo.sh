#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolox_dect
expn=$1
path=$2 # /path/to/you
config=$3
ckpt=$4
cuda=$5
echo -e "\033[31mcuda is ${cuda}\033[0m"
echo -e "\033[31mconfig is ${config}\033[0m"
echo -e "\033[31mckpt is ${ckpt}\033[0m"
echo -e "\033[31mpath is ${path}\033[0m"
echo -e "\033[31mcurrent path is ${PWD}\033[0m"
CUDA_VISIBLE_DEVICES=${cuda} python tools/demo.py image -expn ${expn} -c ${ckpt} \
-f ${config} --path ${path} --fuse  --device cuda # --save_result
cd -


