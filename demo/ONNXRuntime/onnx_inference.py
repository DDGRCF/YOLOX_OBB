#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import cv2
import argparse
import onnxruntime
from yolox.exp import get_exp
from yolox.utils import mkdir, Predictor, DictAction
from yolox.data.data_augment import preproc as preprocess


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--output_format", 
        nargs="+", choices=["bbox", "mask", "obb"], 
        default="bbox")
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="setting some uncertainty values: conf_thre | nms_thre")
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    if len(args.output_format) == 1:
        args.output_format = args.output_format[0]
    exp = get_exp(args.exp_file, None)
    exp.merge(args.options)
    input_names = getattr(exp, "export_input_names", "input")
    output_names = getattr(exp, "export_output_names", "output")
    if not isinstance(input_names, (tuple, list)):
        model_input_names = [input_names]
    if not isinstance(output_names, (tuple, list)):
        model_output_names = [output_names]
    predictor = Predictor(model=None, exp=exp, output_format=args.output_format)
    # input
    ori_image = cv2.imread(args.image_path)
    img, ratio = preprocess(ori_image, exp.test_size)
    img_infos = {"raw_img": ori_image, "ratio": ratio}

    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {input_names[0]: img[None, :, :, :]}
    output = session.run(output_names, ort_inputs)
    res_vis = predictor.visual(output, img_infos)
    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, args.image_path.split("/")[-1])
    cv2.imwrite(output_path, res_vis)
