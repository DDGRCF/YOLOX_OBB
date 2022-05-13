import os
import torch
import argparse
import torch.onnx
import torch.nn as nn
import cv2
import numpy as np
from functools import partial
from yolox.utils import Predictor
from yolox.exp import get_exp
from yolox.utils import replace_module, DictAction, obbpostprocess, postprocess
from yolox.utils import DepolyModel
# from yolox.models.network_blocks import SiLU
from loguru import logger

def make_parser():
    parser = argparse.ArgumentParser()
    # convert type for torchscript / onnx / tensorrt
    parser.add_argument("out_type", choices=["torchscript", "onnx", "tensorrt"])
    # exp config
    parser.add_argument(
        "--output-name", type=str, default="yolox_torchscript", help="output name of models"
    )
    parser.add_argument(
        "--inference-image", type=str, default="images", help="image which help inference"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="device for converting, like cpu / cuda:0")
    # for onnx
    parser.add_argument("--opset-version", type=int, default=11)
    parser.add_argument("--is-onnxsim", action="store_true", default=False)
    parser.add_argument("--workspace-size", type=int, default=32)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="setting some uncertainty values: conf_thre | nms_thre")

    return parser

@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))

    assert args.batch_size is 1, "batch size only support 1"
    assert args.exp_file is not None, "exp file must be provided"

    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.options)
    exp.is_export_onnx = True

    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    device = torch.device(args.device)
    model = exp.get_model()
    # model = replace_module(model, nn.SiLU, SiLU)
    model.to(device)
    ckpt = torch.load(ckpt_file, map_location=device)
    # predictor = Predictor(model, exp, postprocess_func=getattr(model, "postprocess", postprocess), fp16=False, device=device)
    # predictor.inference = partial(predictor.inference, return_img_info=False)

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    logger.info("loading checkpoint done.")
    # input 
    logger.info("begin convert model to torchscript...")
    inference_image_ = cv2.imread(args.inference_image)
    i_h, i_w, c = inference_image_.shape 
    inference_image = np.zeros((*(exp.test_size), c), dtype=np.uint8)
    ratio = min(exp.test_size[0] / i_h, exp.test_size[1] / i_w)
    inference_image_ = cv2.resize(inference_image_, (int(i_w * ratio), int(i_h * ratio)))
    inference_image[:i_h, :i_w, :] = inference_image_
    # dummy_input = (torch.from_numpy(inference_image) / 255).permute(2, 0, 1)[None]
    # dummy_input = np.transpose(inference_image_ / 255.0, (2, 0, 1))
    dummy_input = np.transpose(inference_image, (2, 0, 1))
    # dummy_input -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    # dummy_input /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    dummy_input = torch.from_numpy(dummy_input)[None].to(device).float()
    # dummy_input = torch.rand(1, 3, *(exp.test_size)).to(device)
    # dummy_input = torch.rand(1, 3, 640, 640).to(device)
    depoly_model= DepolyModel(model, exp, max_num=100)
    with torch.jit.optimized_execution(True):
        model_torchscript = torch.jit.trace(depoly_model, dummy_input)
    print(model_torchscript.graph)
    logger.info("torchscript convert done") 
    # if args.out_type == "torchscript" or args.out_type == "onnx" or args.out_type == "tensorrt":
    if args.out_type == "torchscript":
        torchscript_output_path = os.path.join(exp.output_dir, args.output_name + ".pt")
        model_torchscript.save(torchscript_output_path)
        logger.info("generated torchsciopt model named {}".format(args.output_name))
    if args.out_type == "onnx" or args.out_type == "tensorrt":
        onnx_output_path = os.path.join(exp.output_dir, args.output_name + ".onnx")
        input_names = ["input"]
        output_names = ["output"]
        logger.info("onnx model input name is {}".format(input_names))
        logger.info("onnx model output name is {}".format(output_names))

        # generate example output
        dummy_output = model_torchscript(dummy_input)

        # for condinst and boxinst don't support dynamic axes
        logger.info("begin convert onnx model...")
        torch.onnx.export(model_torchscript,
                        dummy_input,
                        onnx_output_path,
                        example_outputs=dummy_output,
                        export_params=True,
                        opset_version=args.opset_version,
                        do_constant_folding=True,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=None,
                        verbose=False)
        logger.info("onnx model convert done.")
        if args.is_onnxsim:
            logger.info("begin simplify onnx modek, and we will check 3 times...")
            import onnx
            from onnxsim import simplify
            input_shapes = {"input": list(dummy_input.shape)}
            onnx_model = onnx.load(onnx_output_path)

            model_simp, check = simplify(onnx_model, check_n = 3, input_shapes=input_shapes, perform_optimization=True)
            assert check, "Simplified ONNX model could not be validate"
            onnx.save(model_simp, onnx_output_path)
            logger.info("simplify onnx model done.")

    if args.out_type == "tensorrt":
        logger.info("begin convert onnx model to tensorrt model...")
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        tensorrt_output_path = os.path.join(exp.output_dir, args.output_name + ".engine")
        with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser, \
            trt.Runtime(TRT_LOGGER) as runtime:
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << args.workspace_size
            with open(onnx_output_path, 'rb') as onnx_model:
                logger.info("begin parsing onnx file")
                if not parser.parse(onnx_model.read()):
                    for error in range(parser.num_errors):
                        logger.info(parser.get_error(error))
                    return -1
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            with open(tensorrt_output_path, "wb") as trt_model:
                trt_model.write(engine.serialize())
            logger.info("begin save trt_model to {}".format(tensorrt_output_path))
    
if __name__ == "__main__":
    main()
