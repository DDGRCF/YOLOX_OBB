#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
import numpy as np
import torch.nn.functional as F
import pycocotools.mask as coco_mask_utils
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, dataloader, 
        img_size, conf_thre=0.1, 
        nms_thre=0.5, num_classes=80, 
        testdev=False, 
        with_bbox=True,
        with_mask=False, 
        metric=["bbox"], 
        save_metric="bbox",
        **kwargs
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        annType = ["bbox", "segm"]
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = conf_thre
        self.nmsthre = nms_thre
        self.num_classes = num_classes
        self.testdev = testdev
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.save_metric = save_metric
        if isinstance(metric, str):
            assert metric in annType
            metric = [metric]
        elif isinstance(metric, list) or isinstance(metric, tuple):
            for m in metric:
                assert m in annType
            metric = set(metric)
        self.metric = metric
        self.kwargs = kwargs

    def evaluate(
        self,
        model,
        is_half=False,
        is_distributed=False,
        decoder=None,
        trt_file=None,
        test_size=None,
        **kwargs
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        self.kwargs.update(kwargs)
        tensor_type = torch.cuda.HalfTensor if is_half else torch.cuda.FloatTensor
        model = model.eval()
        if is_half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                if hasattr(model ,"postprocess"):
                    outputs = model.postprocess(
                        outputs, num_classes=self.num_classes, conf_thre=self.confthre, nms_thre=self.nmsthre, **self.kwargs
                    )
                else:
                    outputs = postprocess(
                        outputs, self.num_classes, conf_thre=self.confthre, nms_thre=self.nmsthre, **self.kwargs
                    )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if is_distributed:
            data_list = gather(data_list, dst=0) 
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        if self.with_bbox and not self.with_mask:
            data_list = self.convert_to_coco_format_bbox(outputs, info_imgs, ids)
        elif not self.with_bbox and self.with_mask:
            data_list =  self.convert_to_coco_format_mask(outputs, info_imgs, ids)
        elif self.with_bbox and self.with_mask:
            data_list = self.convert_to_coco_format_bbox_mask(outputs, info_imgs, ids)
        
        return data_list
        

    def convert_to_coco_format_bbox(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def convert_to_coco_format_mask(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            masks = output[0]
            labels = output[1]
            if masks is None or labels is None:
                continue
            assert labels.shape[-1] > 1
            clses = labels[:, -1]
            if labels.shape[-1]==3 and labels.ndim==2:
                scores = labels[:, 0] * labels[:, 1]
            elif labels.shape[-1]==2 and labels.ndim==2:
                scores = labels[:, 0]
            else:
                raise NotImplementedError
            if masks.shape[1] != self.img_size[0] or masks.shape[2] != self.img_size[1]:
                masks = F.interpolate(masks[:, None], size=(self.img_size[0], self.img_size[1]), 
                                    mode="bilinear", aligne_corners=False).squeeze(1)
            masks = masks[:, :img_h, :img_w] 
            clses = clses.cpu().numpy()
            scores = scores.cpu().numpy()
            masks = masks.cpu().numpy()
            for ind in range(len(scores)):
                label = self.dataloader.dataset.class_ids[int(clses[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": [],
                    "score": scores[ind].item(),
                    "segmentation": []
                }
                seg = coco_mask_utils.encode(
                    np.asarray(masks[ind], order="F", dtype=np.uint8)
                )
                if isinstance(seg["counts"], bytes):
                    seg["counts"] = seg["counts"].decode()
                pred_data.update({"segmentation": seg})
                data_list.append(pred_data)
            return data_list

    def convert_to_coco_format_bbox_mask(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            masks = output[0]
            output = output[1]
            if output is None or masks is None:
                continue

            if masks.shape[1] != self.img_size[0] or masks.shape[2] != self.img_size[1]: 
                masks = F.interpolate(masks[:, None], size=(self.img_size[0], self.img_size[1]), 
                                    mode="bilinear", aligne_corners=False).squeeze(1)
            masks = masks[:, :img_h, :img_w]
            bboxes = output[:, :4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, -1].cpu().numpy()
            if output.shape[-1] == 7:
                scores = (output[:, 4] * output[:, 5]).cpu().numpy()
            elif output.shape[-1] == 6:
                scores = output[:, 4].cpu().numpy()
            masks = masks.cpu().numpy()
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].tolist(),
                    "score": scores[ind].item(),
                    "segmentation": [],
                }  # COCO json format
                seg = coco_mask_utils.encode(
                    np.asarray(masks[ind], order="F", dtype=np.uint8)
                )
                if isinstance(seg["counts"], bytes):
                    seg["counts"] = seg["counts"].decode()
                pred_data.update({"segmentation": seg})
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        eval_stat = {m: [0, 0] for m in self.metric}
        if not is_main_process():
            return eval_stat, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")
            for metric in self.metric:
                info += f"\n*************** Evaluating {metric} ****************\n"
                cocoEval = COCOeval(cocoGt, cocoDt, metric)
                cocoEval.evaluate()
                cocoEval.accumulate()
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                eval_stat[metric] = [cocoEval.stats[0], cocoEval.stats[1]]
                info += redirect_string.getvalue()
            return eval_stat, info
        else:
            return eval_stat, info
