import os
import cv2
import onnx
import torch
import argparse
import numpy as np
from loguru import logger
from yolox.exp import get_exp
from yolox.utils import DictAction

def make_parser():
    parser = argparse.ArgumentParser()
    # convert type for torchscript / onnx / tensorrt
    parser.add_argument("out_type", choices=["torchscript", "onnx", "tensorrt"])
    # exp config
    parser.add_argument(
        "--output-name", type=str, default=None, help="output name of models"
    )
    parser.add_argument(
        "--inference-image", type=str, default="images", help="image which help inference"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
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
    parser.add_argument("--test_model", action="store_true", default=False)
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
    root_expn_dir = "./YOLOX_outputs"

    assert args.batch_size is 1, "batch size only support 1"
    assert args.exp_file is not None, "exp file must be provided"

    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.options)
    model_input_names = getattr(exp, "export_input_names", "input")
    model_output_names = getattr(exp, "export_output_names", "output") 
    dynamic_axes = getattr(exp, "export_dynamic_axes", None)
    if not isinstance(model_input_names, (tuple, list)):
        model_input_names = [model_input_names]
    if not isinstance(model_output_names, (tuple, list)):
        model_output_names = [model_output_names]

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    if not args.output_name:
        args.output_name = exp.exp_name
    
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file_best = os.path.join(file_name, "best_ckpt.pth")
        ckpt_file_last = os.path.join(file_name, "lastest_ckpt.pth")
        if os.path.exists(ckpt_file_best):
            ckpt_file = ckpt_file_best
        elif os.path.exits(ckpt_file_last):
            ckpt_file = ckpt_file_last
        else:
            raise ValueError
    else:
        ckpt_file = args.ckpt

    dst_dir = os.path.join(root_expn_dir, args.experiment_name)
    os.makedirs(dst_dir, exist_ok=True)

    device = torch.device(args.device)
    model = exp.get_model()
    ckpt = torch.load(ckpt_file, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=True)
    logger.info("loading checkpoint done.")
    deploy_model = exp.model_wrapper(model)
    deploy_model.float()
    deploy_model.eval()
    deploy_model.to(device)
    # input 
    inference_image_ = cv2.imread(args.inference_image)
    i_h, i_w, c = inference_image_.shape 
    inference_image = np.zeros((*(exp.test_size), c), dtype=np.uint8)
    ratio = min(exp.test_size[0] / i_h, exp.test_size[1] / i_w)
    inference_image_ = cv2.resize(inference_image_, (int(i_w * ratio), int(i_h * ratio)))
    inference_image[:inference_image_.shape[0], :inference_image_.shape[1] :] = inference_image_
    dummy_input = np.transpose(inference_image, (2, 0, 1))
    dummy_input = torch.from_numpy(dummy_input)[None].to(device).float()
    if args.out_type == "torchscript":
        logger.info("begin convert model to torchscript...")
        with torch.jit.optimized_execution(True):
            deploy_model= torch.jit.trace(deploy_model, dummy_input)
        logger.info("torchscript convert done") 
        torchscript_output_path = os.path.join(dst_dir, args.output_name + ".pt")
        deploy_model.save(torchscript_output_path)
        logger.info("generated torchsciopt model named {}".format(args.output_name))
    if args.out_type == "onnx" or args.out_type == "tensorrt":
        onnx_output_path = os.path.join(dst_dir, args.output_name + ".onnx")
        logger.info("onnx model input name is {}".format(model_input_names))
        logger.info("onnx model output name is {}".format(model_output_names))
        # generate example output
        dummy_output = deploy_model(dummy_input)

        logger.info("begin convert onnx model...")
        torch.onnx.export(deploy_model,
                            dummy_input,
                            onnx_output_path,
                            example_outputs=dummy_output if dynamic_axes is None else None,
                            export_params=True,
                            opset_version=11,#  args.opset_version,
                            do_constant_folding=True,
                            training=torch.onnx.TrainingMode.EVAL,
                            input_names=model_input_names,
                            output_names=model_output_names,
                            dynamic_axes=dynamic_axes,
                            verbose=True)
        logger.info("onnx model convert done.")
        if args.is_onnxsim:
            logger.info("begin simplify onnx modek, and we will check 3 times...")
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

        tensorrt_output_path = os.path.join(dst_dir, args.output_name + ".engine")
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
