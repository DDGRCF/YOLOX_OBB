import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from utils.file_utils import check_dir
from utils.convert_utils import poly2rbox

img_suffixs = ["jpg", "png", "jpeg", "tif"]

if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_dir", type=str, default="")
    parser.add_argument("--dst_dir", type=str, default="")
    args = parser.parse_args()
    ori_dir = Path(args.ori_dir)
    dst_dir = Path(args.dst_dir)
    num = sum(1 for _ in ori_dir.glob("*.json"))
    imagesource = "GoogleEarth"
    gsd = "null"
    check_dir(dst_dir)
    for json_name in tqdm(ori_dir.glob("*.json"), desc="Processing", total=num):
        json_path = Path(json_name)
        json_stem = json_path.stem
        dst_path = dst_dir / (json_stem + ".txt")
        with open(json_path, "r") as fr:
            label_info = json.load(fr)
        shapes = label_info["shapes"]
        with open(dst_path, "w") as fw:
            fw.write(f"imagesource:{imagesource}\n")
            fw.write(f"gsd:{gsd}\n")
            for shape in shapes:
                bbox = shape["points"]
                label = shape["label"]
                new_bbox = []
                for points in bbox:
                    new_bbox.extend(points)
                assert len(new_bbox) == 8, "Only support 4 points"
                dst_info = []
                dst_info.extend(new_bbox)
                dst_info.append(label)
                dst_info.append(0)
                dst_info = list(map(str, dst_info))
                dst_info_str = " ".join(dst_info) + "\n"
                fw.write(dst_info_str)