#! /bin/bash

env=yolox_dect
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${env}

dir=$1
new_dir=$2

python datasets/data_convert_tools/rotated_adjust.py --dir ${dir} --new_dir ${new_dir}
