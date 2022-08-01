import os
import json
import argparse
import pprint
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from utils.file_utils import check_dir
from utils.convert_utils import poly2rbox

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str, default='/path/to/coco/annotations')
    parser.add_argument('--image-source', type=str, default='/path/to/coco/images')
    parser.add_argument('--dst-dir', type=str, default='/dir/to/save/convert_results')
    parser.add_argument('--is_minRect', action='store_true', default=True)
    parser.add_argument('--convert_phase', type=str, choices=['bbox', 'segmentation'], 
                        help='chose convert bbox or segmentation to dota datasets format, \
                            and it can not be set when is_minRect is True')
    parser.add_argument('--is_generate_classes', action='store_true', default=True)
    opt = parser.parse_args()
    assert opt.is_minRect and opt.convert_phase == 'segmentation'
    return opt


@logger.catch
def main(opt):
    json_path = Path(opt.json_path)
    dst_dir = Path(opt.dst_dir)
    image_source = opt.image_source
    is_minRect = opt.is_minRect    
    is_generate_classes = opt.is_generate_classes
    convert_phase = opt.convert_phase
    with open(json_path, 'r') as fr:
        data = json.load(fr)
    check_dir(dst_dir)
    categories = {}
    for category in tqdm(data['categories']):
        cat_id = category['id']
        categories[cat_id] = category['name']
    pprint.pprint(categories)
    if is_generate_classes:
        classes_file_name = dst_dir / '..' / (dst_dir.stem + '_CLASSES.py')
        classes_values = tuple([c for c in categories.values()])
        class_file_content = classes_file_name.stem + ' = ' + str(classes_values)
        with open(classes_file_name, 'w') as fw:
            fw.write(class_file_content)
    for img in tqdm(data['images']):
        filename = Path(img['file_name'])
        img_id = img['id']
        ann_txt = dst_dir / (filename.stem + '.txt')
        with open(ann_txt, 'w') as fw:
            fw.write(f"imagesource:{image_source}\ngsd:null\n")
            for ann in data['annotations']:
                if ann['image_id'] == img_id:
                    if ann.get("iscrowd", False) == True:
                        continue
                    bbox = ann[convert_phase]
                    if is_minRect:
                        bbox = poly2rbox(bbox)
                    bbox = list(map(str, bbox))
                    bbox_text = ' '.join(bbox)
                    ann_text = ' '.join([bbox_text, categories[ann['category_id']], str(0)])  
                    fw.write(ann_text + '\n')

if __name__ == '__main__':
    opt = get_opt()
    main(opt)
