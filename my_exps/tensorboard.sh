#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolox_obb
cd ../
log_dir=$1 # /path/to/work_dir
tensorboard --logdir ${log_dir} --bind_all
