#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolox_obb
cd ../BboxToolkit/tools
class_names=$1
config=$2

python img_split.py --base_json $config --classes $class_names

cd -
