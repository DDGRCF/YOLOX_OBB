import json
import argparse
import numpy as np
import shutil
from tqdm import tqdm

from loguru import logger
from pathlib import Path
from shapely.geometry import *
from utils.convert_utils import *
from utils.file_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="")
    parser.add_argument('--new_dir', type=str, default='')
    parser.add_argument('--use-labels', action='store_true')
    parser.add_argument('--temp-label', type=str, default='null-label')
    args = parser.parse_args()
    use_labels = args.use_labels
    temp_label = args.temp_label
    # check dirs
    assert args.dir!='', "the annotations dir can not be empty"
    dir = Path(args.dir)
    if args.new_dir == '':
        new_dir = dir.parents[0] / 'new_annotations'
    else:
        new_dir = Path(args.new_dir)

    out_boundary = []

    check_dir(new_dir)
    # check label information
    num = sum(1 for _ in dir.glob("*.json"))
    logger.info(f"there are {num} images")
    img_suffixs = ["jpg", "png", "tif", "jpeg"]
    for p in tqdm(dir.glob("*.json"), desc='Processing', total=num):
        with open(p, 'rb') as fr:
            label_info = json.load(fr)
        # copy images to new dir
        for suffix in img_suffixs:
            i_p = p.parents[0] / (p.stem + "." + suffix)
            if i_p.exists():
                break;
        i_p_new = new_dir / i_p.name
        if not i_p_new.exists():
            shutil.copy(i_p, i_p_new)
        # get detail information
        objs = label_info['shapes']
        height = label_info['imageHeight']
        width = label_info['imageWidth']
        new_objs = []
        # get rotated bbox
        for obj in objs:
            if use_labels and obj['label'] != temp_label:
                continue
            points = obj['points']
            line1 = LineString(points[:2])
            point2 = Point(points[2])
            point3 = Point(points[3])
            d1 = point2.distance(line1)
            d2 = point3.distance(line1)
            line2 = line1.parallel_offset(d1, side='right')
            line3 = line1.parallel_offset(d2, side='left')
            side2 = judge_direction(points[0], points[1], points[2]) # judge the direction of the parallel line
            side2_ = judge_direction(points[0], points[1], list(line2.coords[0])) 
            if side2 != side2_:
                line2 = line1.parallel_offset(d1, side='left')
                line3 = line1.parallel_offset(d2, side='right')
            new_points = [list(line2.coords[0]), list(line2.coords[1]), list(line3.coords[0]), list(line3.coords[1])]
            if use_labels:
                get_label(new_points, objs, temp_label)
            else:
                obj['points'] = new_points
        # get images which obj's boundary are out of the images's boundary
        for obj in objs:
            points = obj['points']
            ps = np.array(points)
            if not ((0 <= ps[:, 0]) & (ps[:, 0] <= width - 1) & (0 <= ps[:, 1]) & (ps[:, 1] <= height - 1)).all():
                out_boundary.append(label_info['imagePath'])
            if use_labels:
                if obj['label'] != temp_label:
                    new_objs.append(obj)
        if use_labels:
            label_info['shapes'] = new_objs
        new_p = new_dir / p.name
        with open(new_p, 'w') as fw:
            json.dump(label_info, fw, indent=4)

    with open(new_dir / 'out_boundary.txt', 'w') as fw:
        fw.write(str(out_boundary))


        
    
    
