#! /bin/bash

env=yolox_dect
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

ori_dir=$1
dst_dir=$2
train_name=${3:-train}
val_name=${4:-val}
cls_split_ratio=${5:-3/1}
image_split_ratio=${5:-3/1}

python datasets/data_convert_tools/trainval_split.py ${ori_dir} ${dst_dir} \
--train_set_name ${train_name} --val_set_name ${val_name} --cls_split_ratio ${cls_split_ratio} \
--image_split_ratio ${image_split_ratio}

