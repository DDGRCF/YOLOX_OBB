#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolox_dect

json_path=$1 #/path/to/your/annotations/instances_train2017.json
image_source=$2 #/path/to/your/train2017
dst_dir=$3 #/path/to/save/annotations/instances_train2017_dota

python datasets/data_convert_tools/coco2dota.py --json-path ${json_path} \
--image-source ${image_source} \
--dst-dir ${dst_dir} \
--is_minRect --convert_phase segmentation
