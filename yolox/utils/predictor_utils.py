import torch
import cv2
import os
from yolox.data import ValTransform
from loguru import logger
from time import time
from .visualize import obb_vis, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        result_image = predictor.visual(outputs[0], img_info)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        postprocess_func=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = exp.class_names
        self.decoder = decoder
        self.postprocess_func = postprocess_func
        self.num_classes = exp.num_classes
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.postprocess_cfg = exp.postprocess_cfg
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        if isinstance(self.device, str):
            device = torch.device(self.device)
        else:
            device = self.device

        if device.type != img.device.type:
            img = img.to(device)

        if device.type != self.model.device.type:
            model = model.to(device)

        if device.type == "cuda" and self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = self.postprocess_func(
                outputs, self.num_classes, 
                **self.postprocess_cfg
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))

        return outputs

    def visual(self, output, img_info):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()

        if output.shape[-1] == 11:
            bboxes = output[:, 0:8]
            bboxes /= ratio
            scores = output[:, 8] * output[:, 9]
            cls = output[:, 10]
            vis_res = obb_vis(img, bboxes, scores, cls, class_names=self.cls_names)
        elif output.shape[-1] == 7:
            bboxes = output[:, 0:4]
            bboxes /= ratio
            scores = output[:, 4] * output[:, 5]
            cls = output[:, 6]
            vis_res = vis(img, bboxes, scores, cls, class_names=self.cls_names)

        return vis_res

