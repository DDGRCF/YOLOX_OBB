import os
import cv2 
import json
import pickle
import random
import numpy as np
import os.path as osp
import sys
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from loguru import logger
from .datasets_wrapper import Dataset
from ..dataloading import get_yolox_datadir
from yolox.utils import (time_synchronized, compress_file, poly2obb_np)
from yolox.ops import multiclass_obb_nms
import multiprocessing as mp


class DOTADataset(Dataset):
    
    def __init__(self,
                 name="train",
                 data_dir=None,
                 img_size=(1024, 1024),
                 preproc=None,
                 cache=False,
                 save_results_dir=None):
        super().__init__(img_size)
        if data_dir is None: 
            data_dir = osp.join(get_yolox_datadir(), "DOTA1_0")
        self.imgs = None
        self.name = name
        self.data_dir = data_dir
        self.img_size = img_size
        self.labels_dir = osp.join(data_dir, name, 'annfiles') 
        self.imgs_dir = osp.join(data_dir, name, 'images')
        self.load_dota_infos(self.labels_dir)
        self.preproc = preproc
        self.save_results_dir = save_results_dir
        if cache:
            self._cache_images()
    
    def __len__(self):
        return self.imgs_num
    
    def __del__(self):
        del self.imgs
        # logger.info("Delete the Dota Datasets !!!")

    def load_dota_infos(self, ann_file): 

        split_config_path = osp.join(ann_file, 'split_config.json')
        ori_ann_path = osp.join(ann_file, 'ori_annfile.pkl')
        patch_ann_path = osp.join(ann_file, 'patch_annfile.pkl')
        with open(split_config_path, 'rb') as fr:
            self.split_info = json.load(fr)
        with open(ori_ann_path, 'rb') as fr:
            ori_dict = pickle.load(fr)
        with open(patch_ann_path, 'rb') as fr:
            patch_dict = pickle.load(fr)
        cls, contents = patch_dict['cls'], patch_dict['content']
        ori_cls, ori_contents = ori_dict['cls'], ori_dict['content']
        self.ori_infos = ori_contents
        self.CLASSES = cls if len(ori_cls) == len(cls) else ori_cls
        self.data_infos = contents
        self.imgs_num = len(contents)
        self.ids = [content['id'] for content in contents]
        self.resample_ids = [[] for _ in range(len(self.CLASSES))]
        self.annotations = self._load_dota_annotations()
        # reverse sample: the small class samples have more chances to be sampled
        self.resample_frequency = [(1 / len(i) * 1e3) if len(i) else 0 \
            for i in self.resample_ids]
        

    def load_anno(self, index):
        return self.data_infos[index]['ann']

    def _get_ori_ids(self):
        return list(set([info['ori_id'] for info in self.data_infos]))

    def _load_dota_annotations(self):
        return [self.load_anno_along_ids(index) for index in range(self.imgs_num)]
    
    def load_anno_along_ids(self, index):
        '''
        Return:
        res: the label information of per image, [n_gt_per_img, 9]
        9 contain [8 * polys of dets, labels]
        '''
        im_ann = self.data_infos[index]
        _id = im_ann['id']
        width = im_ann['width']
        height = im_ann['height']
        objs = im_ann['ann']['bboxes'] # shape(n_gt, 8)
        labels = im_ann['ann']['labels'] # shape(n_gt, 1)
        res = np.concatenate((objs, labels[:, None]), axis=-1) # shape(n_gt, 9)
        r = min(self.img_size[0] / width, self.img_size[1] / width)
        res[:, :8] *= r
        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        file_name = (
            im_ann['filename']
            if 'filename' in im_ann
            else "{:012}".format(_id) + '.png'
        )
        # for resample
        for class_id in np.unique(labels):
            self.resample_ids[class_id].append(index)

        return (res, img_info, resized_info, file_name)

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 22G available disk space for training DOTA.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for DOTA"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(self.imgs_num),
            )
            pbar = tqdm(enumerate(loaded_images), total=self.imgs_num)
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(self.imgs_num, max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_file = osp.join(self.imgs_dir, file_name)

        img = cv2.imread(img_file)
        assert img is not None

        return img
    
    def pull_item(self, index):
        _id = self.ids[index]
        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[:resized_info[0], :resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
        
        return img, res.copy(), img_info, _id

    def resample_pull_item(self):
        k = random.choices(range(len(self.CLASSES)), 
                self.resample_frequency, k=1)[0]
        class_ids = self.resample_ids[k]
        index = class_ids[random.randint(0, len(class_ids) - 1)]
        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[:resized_info[0], :resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
        res = res[res[..., -1] == k]
        return img, res.copy(), img_info, index


    @Dataset.wrapper_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h, angle]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): the name of image(like P2021.png, and it is img id will be P2021).
        """
        img, target, img_info, img_id = self.pull_item(index)
        new_target = []
        for t in target:
            new_target.append(poly2obb_np(t))
        if len(new_target) == 0:
            new_target = np.empty((0, 6))
        else:
            new_target = np.concatenate(new_target, axis=0)
        if self.preproc is not None:
            img, new_target = self.preproc(img, new_target, self.input_dim)
        return img, new_target, img_info, img_id
    

    def evaluate_detection(self, 
                           data_list, 
                           merge_nms_thre=0.1,
                           is_submiss=False, 
                           is_merge=False, 
                           save_results_dir=None,
                           eval_func=None,
                           is_merge_nms=True,
                           nproc=10,
                           **kwargs): 
        from functools import partial
        if save_results_dir is not None:
            self.save_results_dir = save_results_dir

        if is_merge:
            xy_start_ids_map = {info["id"]: (info["x_start"], info["y_start"]) for info in self.data_infos}
            ori_id_set = list(set([info['ori_id'] for info in self.data_infos]))
            logger.info("Begin merge, wait...")
            merge_start_time = time_synchronized()
            result_merge_images = [] 
            xy_start_list = []
            for ori_id in ori_id_set:
                xy_start_list_per_image = []
                result_merge_images_per_image = []
                for r in data_list:
                    ori_id_ = r["id"][:r["id"].rfind("_")]
                    if ori_id_ == ori_id:
                        result_merge_images_per_image.append(r)
                        xy_start_list_per_image.append(xy_start_ids_map[r["id"]])
                result_merge_images.append(result_merge_images_per_image)
                xy_start_list.append(xy_start_list_per_image)

            worker = partial(self.single_image_merge, merge_nms_thre=merge_nms_thre, is_merge_nms=is_merge_nms)
            data_bind = list(zip(result_merge_images, xy_start_list, ori_id_set))
            if nproc > 1:
                logger.info("Begin MultiProcess Deal...")
                chunksize=max(len(data_bind) // 10, 1)
                logger.info(f"Attention! We set the chunksize as {chunksize}, you can set it by yourself to get better performance")
                logger.info(f"Change chunksize position : {os.path.realpath(__file__)}:{sys._getframe().f_lineno}")
                data_list = process_map(worker, data_bind, max_workers=nproc, chunksize=chunksize) 
            else:
                logger.info("Begin SingleProcess Deal...")
                data_list = list(map(worker, tqdm(data_bind, total=len(result_merge_images))))

            merge_end_time = time_synchronized()
            logger.info(f"Merge cost time {merge_end_time - merge_start_time:.2f} s")

        if is_submiss:
            from concurrent.futures import ThreadPoolExecutor
            NUM_THREADS = min(8, os.cpu_count()) 
            logger.info("Classify the dets into each cls...")
            dets = []
            for cls_id in tqdm(range(len(self.CLASSES))):
                cls_dets = [(r['id'], 
                            r['scores'][r['labels'] == cls_id], 
                            r['bboxes'][r['labels'] == cls_id]) for r in data_list]
                dets.append(cls_dets)
            if not os.path.exists(self.save_results_dir):
                os.makedirs(self.save_results_dir)
                logger.info(f"Create the dir {self.save_results_dir}")
            logger.info("Begin submiss, wait...")
            submiss_start_time = time_synchronized()
            save_submission_files = partial(self.save_submission_files, output_dir=self.save_results_dir)
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
               executor.map(save_submission_files, range(len(self.CLASSES)), dets) 
            submission_zipfile_path = os.path.join(self.save_results_dir, '..', 'test_results.zip')
            compress_file(submission_zipfile_path, self.save_results_dir)
            submiss_end_time = time_synchronized()
            logger.info(f"Submission cost time {submiss_end_time - submiss_start_time:.2f} s")
            return None, None
        else:
            logger.info(f"Begin eval online, wait...")
            return self._do_eval(data_list, self.ori_infos if is_merge else self.data_infos, eval_func=eval_func)

    def single_image_merge(
        self,
        iters,
        merge_nms_thre=0.1,
        is_merge_nms=True,
    ):
        result_per_image, xy_start_per_image, ori_id  = iters
        new_result_per_image = {}
        bboxes = []
        labels = []
        scores = []
        for result_per_split, xy_start in zip(result_per_image, xy_start_per_image):
            xy_start = np.tile(np.array(xy_start, dtype=result_per_split['bboxes'].dtype), 4)[None]
            bboxes.append(result_per_split['bboxes'] + xy_start)
            labels.append(result_per_split['labels'])
            scores.append(result_per_split['scores'])

        if len(bboxes) > 0:
            bboxes = np.concatenate(bboxes, axis=0)
            labels = np.concatenate(labels, axis=0)
            scores = np.concatenate(scores, axis=0)
        else: 
            bboxes = np.empty((0, 8))
            labels = np.empty((0, ))
            scores = np.empty((0, ))

        if is_merge_nms and len(bboxes):
            mask_nms = multiclass_obb_nms(bboxes, 
                                          scores[..., None], 
                                          labels[..., None], 
                                          iou_thr=merge_nms_thre, 
                                          class_agnostic=False) 
            bboxes = bboxes[mask_nms] 
            labels = labels[mask_nms]
            scores = scores[mask_nms]

        new_result_per_image['id'] = ori_id
        new_result_per_image['bboxes'] = bboxes
        new_result_per_image['labels'] = labels
        new_result_per_image['scores'] = scores
        return new_result_per_image

    def save_submission_files(self, cls_id, cls_results, output_dir=None):
        save_path = osp.join(output_dir, 'Task1_' + self.CLASSES[cls_id] + '.txt')
        with open(save_path, 'w') as fw:
            for result in cls_results:
                img_id = result[0]                 
                scores = result[1]
                bboxes = result[2]
                if len(bboxes) > 0:
                    for i, bbox in enumerate(bboxes):
                        fw.write(
                            ("{} {:.3f}" + " {:.2f}" * 8 + "\n").format(
                                *(img_id, scores[i], *(bbox))
                            )
                        )

    def _do_eval(self, 
                dets, 
                infos,
                metric:str = 'mAP',
                use_07_metric: bool =False, 
                ign_diff=True,
                scale_ranges=None,
                eval_func=None):
        assert metric in ['mAP'], f"Don't support type {metric}"
        assert eval_func is not None, "eval func can't be None"
        id_mapper = {ann['id']: i for i, ann in enumerate(infos)}
        det_results, gt_dst = [], []
        for det in dets:
            det_id = det['id']
            det_bboxes = np.concatenate((det['bboxes'], det['scores'][..., None]), axis=-1)
            det_labels = det['labels']
            det_results.append([det_bboxes[det_labels == i] for i in range(len(self.CLASSES))]) 
            ann = infos[id_mapper[det_id]]['ann']
            gt_bboxes = ann['bboxes']
            gt_labels = ann['labels']
            diffs = ann.get('diffs', np.zeros((gt_bboxes.shape[0], ), dtype=np.int))
            # TODO: support Task2 
            gt_ann = {}
            if ign_diff:
                gt_ann['bboxes_ignore'] = gt_bboxes[diffs == 1]
                gt_ann['labels_ignore'] = gt_labels[diffs == 1]
                gt_bboxes = gt_bboxes[diffs == 0]
                gt_labels = gt_labels[diffs == 0]
            gt_ann['bboxes'] = gt_bboxes
            gt_ann['labels'] = gt_labels
            gt_dst.append(gt_ann)
        if metric == 'mAP':
            IouTh = np.linspace(
                0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
            )
            mAPs = []
            for iou in IouTh:
                mAP = eval_func(det_results, gt_dst, self.CLASSES, scale_ranges, iou, use_07_metric=use_07_metric, nproc=4)[0]
                mAPs.append(mAP)

            msg = f"map_5095: {np.mean(mAPs)} | map_50: {mAPs[0]}"
            logger.info("\n" + "-" * len(msg) + "\n" + msg + "\n" + "-" * len(msg))
 
            return np.mean(mAPs), mAPs[0]
