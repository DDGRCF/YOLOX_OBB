# dir - images
# dir - labels
import os
import json
import glob
import argparse
import time
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from utils.file_utils import check_dir

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ori_dir", type=str, default="", help="")
    parser.add_argument("dst_dir", type=str, default="", help="")
    parser.add_argument("--train_set_name", type=str, default=None, help="")
    parser.add_argument("--val_set_name", type=str, default=None, help="")
    parser.add_argument("--ignore_classes", nargs="+", help="")
    parser.add_argument("--image_split_ratio", type=float, default=3/1, help="train image num / val image num")
    parser.add_argument("--cls_split_ratio", type=float, default=3/1, help="train cls obj num / val cls obj num")
    parser.add_argument("--cls_split_ratio_range", type=float, default=1/5, help="")
    parser.add_argument("--data_type", type=str, choices=["coco", "dota"], help="")
    parser.add_argument("--del_collect_files", action="store_false", help="")
    parser.add_argument("--image_suffix", type=str, default="tif")
    parser.add_argument("--label_suffix", type=str, default="txt")
    args = parser.parse_args()
    if args.train_set_name is None or args.val_set_name is None:
        local_time = time.asctime(time.localtime(time.time()))
        local_time = local_time.split(" ")
        local_time = "_".join(local_time)
    if args.train_set_name is None:
        args.train_set_name = "train_split" + "-" + local_time
    if args.val_set_name is None:
        args.val_set_name = "val_split" + "-" + local_time

    return args

def read_coco_data():
    pass

def read_dota_data(data_dir, 
                   image_dir_name="images", 
                   label_dir_name="labels", 
                   label_suffix="txt", 
                   image_suffix="tif",
                   del_files=False):

    logger.info("Begin read dota infos...")
    per_cls_stems = {}
    per_image_cls_nums = {}
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    temp_file_path = data_dir / "collect_infos.json"
    if temp_file_path.exists() or not del_files:
        logger.info(f"Find collect file {temp_file_path}")
        with open(temp_file_path, "rb") as fr:
            collect_infos = json.load(fr)
        per_cls_stems = collect_infos["per_cls_stems"]
        per_image_cls_nums = collect_infos["per_image_cls_nums"]
    else:
        logger.info(f"If exists, del collect file {temp_file_path}")
        image_dir = data_dir / image_dir_name
        label_dir = data_dir / label_dir_name
        glob_label_files = list(label_dir.glob("*." + label_suffix))
        for label_file_path in tqdm(glob_label_files):
            file_stem = label_file_path.stem 
            label_file_name = label_file_path.name
            image_file_name = file_stem + "." + image_suffix
            image_file_path = image_dir / image_file_name
            with open(label_file_path, "r") as fr:
                lines = fr.readlines()
            for line in lines:
                line = line.strip("\n")
                line = line.lstrip(" ").strip(" ")
                if line.startswith("imagesource") or line.startswith("gsd"):
                    continue
                items = line.split(" ")
                if len(items) < 9:
                    raise ValueError
                class_name = items[8]
                if class_name in per_cls_stems:
                    per_cls_stems[class_name].append(file_stem)
                else:
                    per_cls_stems[class_name] = [file_stem]
                if file_stem in per_image_cls_nums:
                    if class_name not in per_image_cls_nums[file_stem]:
                        per_image_cls_nums[file_stem].update({class_name: 1})
                    per_image_cls_nums[file_stem][class_name] += 1
                else:
                    per_image_cls_nums[file_stem] = {class_name: 1}
            for cls, cls_stems in per_cls_stems.items():
                per_cls_stems[cls] = list(set(cls_stems)) 

        collect_infos = dict(per_cls_stems=per_cls_stems, per_image_cls_nums=per_image_cls_nums)
        with open(temp_file_path, "w") as fw:
            json.dump(collect_infos, fw)
        logger.info(f"Save collect_infos into {temp_file_path}")

    return per_cls_stems, per_image_cls_nums

def trainval_split(
    ori_dir="./",
    dst_dir="./",
    train_set_name="",
    val_set_name="",
    ignore_classes="",
    image_split_ratio=3/1, 
    cls_split_ratio=3/1,
    cls_split_ratio_range=1/5,
    data_type="",
    del_collect_files=False,
    image_suffix="tif",
    label_suffix="txt"
):
    if data_type == "coco":
        per_cls_stems, per_image_cls_nums = read_coco_data(ori_dir, del_files=del_collect_files)
    elif data_type == "dota":
        per_cls_stems, per_image_cls_nums = read_dota_data(
            ori_dir, del_files=del_collect_files, 
            label_suffix=label_suffix, image_suffix=image_suffix
        )
    
    info_msg = "Pop classes:"
    warn_msg = "Miss pop class:"
    for classes in ignore_classes:
        if per_cls_stems.pop(classes, None) is not None:
            info_msg = info_msg + " " + classes
        else:
            warn_msg = warn_msg + " " + classes 
    if len(warn_msg) - warn_msg.rfind(":") > 1:
        logger.warning(warn_msg)
    if len(info_msg) - info_msg.rfind(":") > 1:
        logger.info(info_msg)
    evolution = True
    # per_image_cls_nums = {k: v for k, v in sorted(per_image_cls_nums.items(), key=lambda items : len(items[1]))}
    per_cls_stems = {k: v for k, v in sorted(per_cls_stems.items(), key=lambda items : len(items[1]))}
    if not isinstance(ori_dir, Path):
        ori_dir = Path(ori_dir)
        dst_dir = Path(dst_dir)

    cls_ratio_range = (cls_split_ratio - cls_split_ratio * cls_split_ratio_range, 
                       cls_split_ratio + cls_split_ratio * cls_split_ratio_range)
    logger.info("Split image...")
    while evolution:
        train_set_list = []
        val_set_list = []
        for cls, cls_objs in per_cls_stems.items():
            cls_objs = [obj for obj in cls_objs if obj not in train_set_list + val_set_list]
            cls_num = len(cls_objs) 
            val_num = max(int(cls_num * 1 / image_split_ratio), 1)
            train_num = cls_num - val_num
            cls_val_objs = random.sample(cls_objs, k=val_num)
            cls_train_objs = [obj for obj in cls_objs if obj not in cls_val_objs]
            assert len(cls_val_objs) == val_num and len(cls_train_objs) == train_num
            val_set_list.extend(cls_val_objs)
            train_set_list.extend(cls_train_objs)
            logger.info(f"class: {cls} train_num: {train_num} val_num: {val_num} ratio: {train_num / val_num:.3f}")

        train_val_per_cls_ratio = {k: {"train": 0, "val": 0} for k in per_cls_stems}
        for cls in train_val_per_cls_ratio:
            for t_o in train_set_list: 
                train_val_per_cls_ratio[cls]["train"] += per_image_cls_nums[t_o].get(cls, 0)
            for t_o in val_set_list:
                train_val_per_cls_ratio[cls]["val"] += per_image_cls_nums[t_o].get(cls, 0)
        logger.info("\n")
        evolution = False
        for cls, value in train_val_per_cls_ratio.items():
            ratio = value["train"] / value["val"]
            msg = "class: {} train_num: {} val_num: {} ratio: {:.3f}".format(cls, value["train"], value["val"], ratio)
            logger.info(msg)
            if ratio < min(cls_ratio_range) or ratio > max(cls_ratio_range):
                evolution = True
                break
        logger.info("---------------------------------------------------------------")
    
    assert len(set(train_set_list) & set(val_set_list)) == 0
    

    logger.info("Save to images...")
    train_set_dir = dst_dir / train_set_name
    val_set_dir = dst_dir / val_set_name
    if data_type == "dota":
        train_set_image_dir = train_set_dir / "images"
        train_set_label_dir = train_set_dir / "labelTxt"
        check_dir(train_set_image_dir, del_exist=True, print_info=True)
        check_dir(train_set_label_dir, del_exist=True, print_info=True)
        logger.info("Train set save...")
        for obj in tqdm(train_set_list):
            ori_train_set_image_path = ori_dir / "images" / (obj + "." + image_suffix)
            ori_train_set_label_path = ori_dir / "labels" / (obj + "." + label_suffix)
            train_set_image_path= train_set_image_dir / (obj + "." + image_suffix)
            train_set_label_path = train_set_label_dir / (obj + "." + label_suffix)
            shutil.copyfile(ori_train_set_image_path, train_set_image_path)
            shutil.copyfile(ori_train_set_label_path, train_set_label_path)

        val_set_image_dir = val_set_dir / "images"
        val_set_label_dir = val_set_dir / "labelTxt"
        check_dir(val_set_image_dir, del_exist=True, print_info=True)
        check_dir(val_set_label_dir, del_exist=True, print_info=True)
        logger.info("Val set save...")
        for obj in tqdm(val_set_list):
            ori_val_set_image_path = ori_dir / "images" / (obj + "." + image_suffix)
            ori_val_set_label_path = ori_dir / "labels" / (obj + "." + label_suffix)
            val_set_image_path= val_set_image_dir / (obj + "." + image_suffix)
            val_set_label_path = val_set_label_dir / (obj + "." + label_suffix)
            shutil.copyfile(ori_val_set_image_path, val_set_image_path)
            shutil.copyfile(ori_val_set_label_path, val_set_label_path)


if __name__ == "__main__":
    args = get_args()
    trainval_split(**vars(args))