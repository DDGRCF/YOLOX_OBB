#! /bin/bash

env=yolox_dect
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

ori_dir=$1
dst_dir=$2

python datasets/data_convert_tools/labelme2dota.py --ori_dir ${ori_dir} --dst_dir ${dst_dir}
