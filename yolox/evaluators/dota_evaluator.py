#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import time
import torch
import itertools
import numpy as np
from loguru import logger
from tqdm import tqdm
from .dota_eval import eval_arb_map


from yolox.utils import (
    gather,
    is_main_process,
    obbpostprocess,
    synchronize,
    time_synchronized,
)

class DOTAEvaluator:
    """
    DOTA AP Evaluation class.  
    """

    def __init__(
        self, dataloader, img_size, conf_thre, nms_thre, num_classes, testdev=False, ign_diff=0, **kwargs
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
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = conf_thre
        self.nmsthre = nms_thre
        self.num_classes = num_classes
        self.testdev = testdev
        self.ign_diff = ign_diff

    def evaluate(
        self,
        model,
        is_half=False,
        is_distributed=False,
        is_merge=False,
        is_submiss=False,
        decoder=None,
        trt_file=None,
        test_size=None,
        **kwargs
    ):
        """
        DOTA average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by DOTAAPI.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
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
        # if trt_file is not None:
        #     from torch2trt import TRTModule

        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))

        #     x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
        #     model(x)
        #     model = model_trt
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

                outputs = obbpostprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre, False
                ) # shape(n_pre, 10)
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(self.convert_to_dota_format(outputs, info_imgs, ids))
        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if is_distributed:
            data_list = gather(data_list, dst=0)
            data_list = itertools.chain(*data_list)
            torch.distributed.reduce(statistics, dst=0)
        eval_results = self.evaluate_prediction(data_list, statistics, is_submiss, is_merge, **kwargs)
        synchronize()
        return  eval_results

    def convert_to_dota_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                pred_data = {
                    "id": img_id,
                    "bboxes": np.empty((0, 8), dtype=np.float32),
                    "labels": np.empty((0, ), dtype=np.float32),
                    "scores": np.empty((0, ), dtype=np.float32),
                }
                data_list.append(pred_data)
                continue
            output = output.cpu()

            bboxes = output[:, :8]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale

            cls = output[:, 10]
            scores = output[:, 8] * output[:, 9]
            pred_data = {
                "id": img_id,
                "bboxes": bboxes.float().numpy(),
                "labels": cls.float().numpy(),
                "scores": scores.float().numpy()
            }
            data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_list, statistics, is_submiss=False, is_merge=True, **kwargs):
        eval_stat = {"bbox": [0, 0]}
        if not is_main_process():
            return 0, 0, None
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
        eval_func = kwargs.pop("eval_func", eval_arb_map)
        if isinstance(eval_func, str):
            try:
                eval_func = eval(eval_func)
            except:
                logger.warning(f"Can't find eval func: {eval_func}, \
                               will set {eval_arb_map.__name__} as default eval func")

        mAPs, mAP50 = self.dataloader.dataset.evaluate_detection(data_list, 
                                                                 merge_nms_thre=kwargs.pop("merge_nms_thre", 0.1),
                                                                 is_submiss=is_submiss,
                                                                 is_merge=is_merge,
                                                                 eval_func=eval_func,
                                                                 **kwargs) 
        if mAPs is not None and mAP50 is not None:
            eval_stat["bbox"] = [mAPs, mAP50]

        if mAPs is None or mAP50 is None:
            return eval_stat, None
        else:
            return eval_stat, info