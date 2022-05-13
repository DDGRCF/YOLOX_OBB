#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolox_obb
cd ../datasets/data_convert_tools
json_path=/path/to/your/annotations #/path/to/your/annotations/instances_train2017.json
image_source=path/to/your/images/dir/ #/path/to/your/train2017
dst_dir=/path/to/your/save/dir #/path/to/save/annotations/instances_train2017_dota
python coco2dota.py --json-path ${json_path} \
--image-source ${image_source} \
--dst-dir ${dst_dir} \
--is_minRect --convert_phase segmentation