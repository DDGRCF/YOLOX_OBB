#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger
from torch import is_distributed
from yolox.exp import OBBExp as MyExp
from yolox.utils.ema import is_parallel

class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 80
        self.no_aug_epochs = 2
        self.no_eval = True
        self.mosaic_prob = 1.0
        self.copy_paste_prob = 1.0
        self.mixup_prob = 0.0
        self.enable_resample = True # for resampling samples
        # enable debug which allow usr to debug aug images
        self.enable_debug = False
        # ignore images which exists horizontal labels, 
        # the rotated aug will not implement the classes by adding this item
        self.aug_ignore = ['roundabout', 'storage-tank'] 
        # ignore images which has no labels, which ensure each train contains labels
        self.empty_ignore = True
        self.evaluate_cfg = dict(
            is_submiss=False,
            is_merge=False,
            nproc=10)

        self.postprocess_cfg = dict(
            conf_thre=0.05,
            nms_thre=0.1,
        )

        self.vssp_cfg = dict(
            sp_num_process=10,
            sp_test_ckpt_path="YOLOX_outputs/dota1_new/latest_ckpt.pth",
            sp_tpfp_func="tpfp_default",
            sp_iou_thre=0.1,
            sp_batch_size=8,
            sp_is_merge=False,
            sp_is_submiss=True,
            sp_is_fuse=True,
            sp_is_fp16=True,
            sp_is_resample=True,
            sp_postprocess_cfg = dict(
                conf_thre=0.05,
                nms_thre=0.1,
                class_agnostic=True
            )
        )

    def _get_sp_eval_loader(self, batch_size, save_results_dir):
        import torch
        from yolox.data import DOTADataset, ValTransform

        valdataset = DOTADataset(
            data_dir=self.data_dir,
            name=self.train_ann,
            img_size=self.test_size,
            preproc=ValTransform(legacy=False),
            save_results_dir=save_results_dir
        )
        sampler = torch.utils.data.SequentialSampler(valdataset)
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader


    def _get_sp_eval_evaluator(self, batch_size, save_results_dir):
        from yolox.evaluators import DOTAEvaluator

        val_loader = self._get_sp_eval_loader(batch_size, save_results_dir)
        evaluator = DOTAEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            num_classes=self.num_classes,
            testdev=False,
            **getattr(self.vssp_cfg, "sp_postprocess_cfg", self.postprocess_cfg)
        )

        return evaluator

    def _do_sp_eval(self, model, ckpt_file, 
                    save_results_dir, batch_size=1, 
                    is_fuse=True, is_fp16=False, rank=0, **kwargs):
        import torch
        evaluator = self._get_sp_eval_evaluator(batch_size, save_results_dir)
        torch.cuda.set_device(rank)
        model.cuda(rank)
        model.eval()
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt.get("model", ckpt))
        logger.info("Load checkpoint done.")
        *_, _ = evaluator.evaluate(
            model, is_half=is_fp16,
            is_distributed=False,
            decoder=None,
            trt_file=None,
            test_size=self.test_size,
            **kwargs
        )
    
    def _get_data_info(self, data_config):
        import os
        import json
        import pickle
        import numpy as np
        import os.path as osp
        import BboxToolkit as bt
        from tqdm import tqdm
        from glob import glob
        from copy import deepcopy
        from multiprocessing import Pool
        from tempfile import TemporaryDirectory
        from yolox.evaluators.dota_eval import tpfp_default
        from yolox.data import DOTADataset, ValTransform
        try:
            sp_tmp_dir = TemporaryDirectory()
            sp_batch_size = self.vssp_cfg.get("sp_batch_size", 1)
            sp_test_ckpt_path = self.vssp_cfg.get("sp_test_ckpt_path")
            sp_is_merge = self.vssp_cfg.get("sp_is_merge", False)
            sp_is_submiss = self.vssp_cfg.get("sp_is_submiss", True)
            sp_is_fuse = self.vssp_cfg.get("sp_is_fuse", True)
            sp_is_fp16 = self.vssp_cfg.get("sp_is_fp16", False)
            sp_is_resample = self.vssp_cfg.get("sp_is_resample", False)
            results_path = sp_tmp_dir.name

            logger.info("Get data infos...")
            super()._get_data_info(data_config)
            dataset = DOTADataset(
                data_dir=self.data_dir,
                name=self.train_ann,
                img_size=self.test_size,
                preproc=ValTransform(legacy=False)
            )
            logger.info("Create tmp model...")
            ori_model = self.get_model()
            delattr(self, "model")
            logger.info("Begin evaluating the train dataset...")
            self._do_sp_eval(ori_model,
                                sp_test_ckpt_path,
                                sp_tmp_dir.name, sp_batch_size,
                                is_fuse=sp_is_fuse, is_fp=sp_is_fp16,
                                is_merge=sp_is_merge, is_submiss=sp_is_submiss)
            logger.info("Begin collecting det and ann infos...")
            gt_infos = dataset.data_infos
            infos_id_map = {infos["id"]: infos for infos in gt_infos}

            def get_det_results():
                results = {}
                logger.info("Load cls det infos...")
                for per_cls_txt_file in tqdm(glob(osp.join(results_path, "*.txt"))):
                    with open(per_cls_txt_file, "r") as fr:
                        for line in fr.readlines():
                            line = line.strip("\n")
                            line = line.split(" ")
                            img_id = line[0]
                            prob = float(line[1])
                            bbox = list(map(float, line[2:]))
                            bbox.append(prob)
                            bbox = np.asarray(bbox, dtype=np.float32)[None]
                            if img_id in results:
                                results[img_id] = np.concatenate((results[img_id], bbox))
                            else:
                                results[img_id] = bbox
                return results
            det_results = get_det_results()

            infos_id_map = {ann["id"]: ann for ann in gt_infos}
            ann_bbox_type = bt.get_bbox_type(gt_infos[0]["ann"]["bboxes"])

            def get_ann_results():
                results = {}
                logger.info("Load cls ann infos...")
                for img_id, bboxes in tqdm(det_results.items()):
                    ann = infos_id_map[img_id]["ann"]
                    bboxes = ann["bboxes"]
                    results[img_id] = bboxes
                return results

            ann_results = get_ann_results()
            tpfp_func = self.vssp_cfg.get('sp_tpfp_func', tpfp_default)
            if isinstance(tpfp_func, str):
                try:
                    tpfp_func = eval(tpfp_func)
                except:
                    logger.warning(f"Parse {tpfp_func} failed! We will set {tpfp_default.__name__} as tpfp_func!")
                    tpfp_func = tpfp_default

            logger.info("Begin multiprocess-calculating the fp objects...")
            num_process = self.vssp_cfg.get("sp_num_process", 10)
            pool = Pool(num_process)
            new_gt_infos = deepcopy(gt_infos)
            new_infos_id_map = {ann["id"]: ann for ann in new_gt_infos}
            num_det_results = len(det_results)
            gt_bboxes_ignore = [np.empty((0, bt.get_bbox_dim(ann_bbox_type))) for _ in range(num_det_results)]
            iou_thr_list = [self.vssp_cfg.get("sp_iou_thre", 0.5) for _ in range(num_det_results)]
            area_ranges = [None for _ in range(num_det_results)]
            tpfp = pool.starmap(tpfp_func, zip(list(det_results.values()), 
                                    list(ann_results.values()), gt_bboxes_ignore, iou_thr_list, area_ranges))
            _, fp_list = tuple(zip(*tpfp))
            for img_id, fp, per_img_det_results, per_img_ann_results in tqdm(zip(det_results.keys(), fp_list, det_results.values(), ann_results.values())):
                fp = fp.squeeze(0).astype(np.bool8)
                ori_bboxes = per_img_det_results[:, :-1]
                fp_bboxes = ori_bboxes[fp]
                fp_labels = np.full((len(fp_bboxes), ), self.num_classes, dtype=np.int64)
                fp_diffs = np.zeros((len(fp_labels), ), dtype=np.int64)
                fp_trunc = np.zeros((len(fp_labels), ), dtype=np.bool8)
                ann = new_infos_id_map[img_id]["ann"]
                if len(fp_bboxes):
                    ann["bboxes"] = np.concatenate((ann["bboxes"], fp_bboxes), axis=0, dtype=np.float32)
                    ann["labels"] = np.concatenate((ann["labels"], fp_labels), axis=0, dtype=np.int64)
                    ann["diffs"] = np.concatenate((ann["diffs"], fp_diffs), axis=0, dtype=np.int64)
                    ann["trunc"] = np.concatenate((ann["trunc"], fp_trunc), axis=0, dtype=np.bool8)
            
            new_cls = list(dataset.CLASSES)
            new_cls.append("null")
            new_cls = tuple(new_cls)
            new_patch_annfile_dict = {"cls": new_cls, "content": new_gt_infos}
            self.class_names = new_cls
            self.num_classes = len(new_cls)
            def load_dota_infos(ctx, ann_file,
                                new_patch_annfile=deepcopy(new_patch_annfile_dict),
                                is_resample=sp_is_resample and self.copy_paste_prob > 0.0 and self.enable_resample):
                split_config_path = osp.join(ann_file, 'split_config.json')
                ori_ann_path = osp.join(ann_file, 'ori_annfile.pkl')
                with open(split_config_path, "rb") as fr:
                    ctx.split_info = json.load(fr)
                with open(ori_ann_path, "rb") as fr:
                    ori_dict = pickle.load(fr)
                if "train" in ctx.name:
                    patch_dict = new_patch_annfile
                else:
                    patch_ann_path = osp.join(ann_file, "patch_annfile.pkl")
                    with open(patch_ann_path, "rb") as fr:
                        patch_dict = pickle.load(fr)
                cls, content = patch_dict["cls"], patch_dict["content"]
                ori_cls, ori_content = ori_dict["cls"], ori_dict["content"]
                ctx.ori_infos = ori_content
                if ctx.name == self.train_ann:
                    ctx.CLASSES = cls
                else:
                    ctx.CLASSES = cls if len(ori_cls) == len(cls) else ori_cls
                ctx.data_infos = content
                ctx.imgs_num = len(content)
                ctx.ids = [c["id"] for c in content]
                ctx.resample_ids = [[] for _ in range(len(ctx.CLASSES))]
                ctx.annotations = ctx._load_dota_annotations()
                ctx.resample_frequency = [(1 / len(ids) * 1e3) if len(ids) else 0 \
                    for i, ids in enumerate(ctx.resample_ids)]
                if "train" in ctx.name:
                    ctx.resample_frequency[-1] = 0
            
            DOTADataset.load_dota_infos = load_dota_infos

            from yolox.core.trainer import Trainer
            from yolox.utils.checkpoint import save_checkpoint
            from yolox.utils.dist import synchronize

            def evaluate_and_save_model(ctx, ori_model=deepcopy(ori_model)):
                if ctx.use_model_ema:
                    evalmodel = ctx.ema_model.ema
                else:
                    evalmodel = ctx.model
                    if is_parallel(evalmodel):
                        evalmodel = evalmodel.module
                logger.info("Pruning redundancy parameters of the model...")
                new_ori_model_state_dict = {}
                ori_model_state_dict = ori_model.state_dict()
                after_model_state_dict = evalmodel.state_dict()
                for ori_model_params_name, after_model_params_name in zip(ori_model_state_dict, after_model_state_dict):
                    assert ori_model_params_name == after_model_params_name
                    ori_model_params = ori_model_state_dict[ori_model_params_name]
                    after_model_params = after_model_state_dict[after_model_params_name]
                    ori_shape = ori_model_params.shape
                    after_shape = after_model_params.shape
                    if ori_shape == after_shape:
                        new_ori_model_state_dict.update({ori_model_params_name: after_model_params})
                    else:
                        dims = after_model_params.dim()
                        logger.info("Name: {} | Convert Shape from {} to {}".format(ori_model_params_name, list(after_shape), list(ori_shape)))
                        if dims == 4:
                            new_ori_model_state_dict.update({ori_model_params_name: after_model_params[:ori_shape[0], :ori_shape[1], :ori_shape[2], :ori_shape[3]]})
                        elif dims == 3:
                            new_ori_model_state_dict.update({ori_model_params_name: after_model_params[:ori_shape[0], :ori_shape[1], :ori_shape[2]]})
                        elif dims == 2:
                            new_ori_model_state_dict.update({ori_model_params_name: after_model_params[:ori_shape[0], :ori_shape[1]]})
                        elif dims == 1:
                            new_ori_model_state_dict.update({ori_model_params_name: after_model_params[:ori_shape[0]]})
                        else:
                            logger.error("Error Name: {} | Shape: {}".format(ori_model_params_name, list(ori_shape)))
                            raise NotImplementedError
                ori_model.to(next(evalmodel.parameters()).dtype).to(next(evalmodel.parameters()).device)
                ori_model.load_state_dict(new_ori_model_state_dict)
                ori_model.eval()
                eval_dict, summary = ctx.exp.eval(
                    ori_model, ctx.evaluator, ctx.is_distributed
                )
                ctx.model.train()
                if ctx.rank == 0:
                    for metric, eval_stat in eval_dict.items():
                        ap50, ap50_95 = eval_stat
                        ctx.tblogger.add_scalar(f"{metric}_val/AP50", ap50, ctx.epoch + 1)
                        ctx.tblogger.add_scalar(f"{metric}_val/AP50_95", ap50_95, ctx.epoch + 1)
                    logger.info("\n" + summary)
                synchronize()

                ctx.save_ckpt("last_epoch", eval_dict[ctx.save_metric][1] > ctx.best_ap, deepcopy(ori_model))
                ctx.best_ap = max(ctx.best_ap, eval_dict[ctx.save_metric][1])
                        
                
            def save_ckpt(ctx, ckpt_name, update_best_ckpt=False, pruning_model=None):
                if ctx.rank == 0:
                    save_model = ctx.ema_model.ema if ctx.use_model_ema else ctx.model
                    ckpt_state = {
                        "start_epoch": ctx.epoch + 1,
                        "model": save_model.state_dict(),
                        "optimizer": ctx.optimizer.state_dict(),
                    }
                    if pruning_model is not None:
                        new_ckpt_state = deepcopy(ckpt_state)
                        new_ckpt_state["model"] = pruning_model.state_dict()
                        new_save_path = osp.join(ctx.file_name, "real_ckpts")
                        if not osp.exists(new_save_path):
                            os.makedirs(new_save_path)
                        save_checkpoint(
                            new_ckpt_state,
                            update_best_ckpt,
                            new_save_path,
                            ckpt_name,
                        )

                    save_checkpoint(
                        ckpt_state,
                        update_best_ckpt,
                        ctx.file_name,
                        ckpt_name,
                    )
            Trainer.save_ckpt = save_ckpt
            Trainer.evaluate_and_save_model = evaluate_and_save_model
        finally:
            sp_tmp_dir.cleanup()

        