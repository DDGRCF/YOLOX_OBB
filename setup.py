#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

import re
import setuptools
import glob
import os
from os import path
import torch
import warnings
from torch.utils.cpp_extension import (CppExtension, CUDAExtension)

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 7], "Requires PyTorch >= 1.7"

def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "yolox", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "yolox._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
        make_cuda_ext(
            name = 'box_iou_rotated_ext',
            module = 'yolox.ops.pytorch.box_iou_rotated', 
            sources = [
                'src/box_iou_rotated_cpu.cpp',
                'src/box_iou_rotated_ext.cpp'
            ],
            sources_cuda = ['src/box_iou_rotated_cuda.cu']
        ),
        make_cuda_ext(
            name = 'convex_ext',
            module = 'yolox.ops.pytorch.convex',
            sources = [
                'src/convex_cpu.cpp',
                'src/convex_ext.cpp'
            ],
            sources_cuda = ['src/convex_cuda.cu']
        ),
        make_cuda_ext(
            name = 'nms_rotated_ext',
            module = 'yolox.ops.pytorch.nms_rotated',
            sources = [
                'src/nms_rotated_cpu.cpp',
                'src/nms_rotated_ext.cpp',
            ],
            sources_cuda = [
                'src/nms_rotated_cuda.cu',
                'src/poly_nms_cuda.cu']
        ),
        make_cuda_ext(
            name = 'roi_align_ext',
            module = 'yolox.ops.pytorch.roi_align',
            sources=[
                'src/roi_align_ext.cpp',
                'src/cpu/roi_align_v2.cpp',
            ],
            sources_cuda=[
                'src/cuda/roi_align_kernel.cu',
                'src/cuda/roi_align_kernel_v2.cu'
        ]),
        make_cuda_ext(
            name='roi_align_rotated_ext',
            module='yolox.ops.pytorch.roi_align_rotated',
            sources=[
                'src/roi_align_rotated_cpu.cpp',
                'src/roi_align_rotated_ext.cpp'
            ],
            sources_cuda=['src/roi_align_rotated_cuda.cu']),
    ]
    if os.getenv("ONNXRUNTIME_DIR", "0") != "0":
        warnings.warn("This Part is incompleted!")
        ext_modules.append(
            make_onnxruntime_ext(
                name="ort_ext",
                module="yolox.ops.onnxruntime",
                with_cuda=False
            )
        )

    return ext_modules

def make_cuda_ext(name, module, sources, sources_cuda=[], with_cuda=True):
    define_macros = []
    extra_compile_args = {'cxx': ["-O2", "-std=c++14", "-Wall"]}
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1' and with_cuda:
        define_macros += [('WITH_CUDA', None)] # 宏定义的声明，#define WITH_CUDA None
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-O2',
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f"Compiling {name} without CUAD")
        extension = CppExtension
    return extension(
        name=f"{module}.{name}",
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compilr_args=extra_compile_args
    )

def make_onnxruntime_ext(name, module, with_cuda=True):
    source_type = ("cpp", "c", "cc")
    include_type = ("hpp", "h", "cuh")
    source_files = []
    include_dirs = []
    define_macros = []
    module_path = os.path.join(*module.split("."))
    extra_compile_args = {'cxx': ["-O2", "-std=c++14", "-Wall"]}
    for t in source_type:
        for p in glob.glob(module_path + "/**/*." + t, recursive=True):
            source_files.append(p)
    for t in include_type:
        for p in glob.glob(module_path + "/**/*." + t, recursive=True):
            include_dirs.append(os.path.abspath(os.path.dirname(p)))
    include_dirs = list(set(include_dirs))
    if (torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1') and with_cuda:
        define_macros += [('WITH_CUDA', None)] # 宏定义的声明，#define WITH_CUDA None
        raise NotImplementedError
    else:
        ort_path = os.path.abspath(os.getenv("ONNXRUNTIME_DIR", "0"))
        libraries = ["onnxruntime"]
        library_dirs = [os.path.join(ort_path, "lib")]
        include_dirs.append(os.path.join(ort_path, "include"))
        extension = CppExtension
        return extension(
            name=f"{module}.{name}",
            sources=source_files,
            define_macros=define_macros,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            libraries=libraries,
            library_dirs=library_dirs)


with open("yolox/__init__.py", "r") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(), re.MULTILINE
    ).group(1)


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setuptools.setup(
    name="yolox",
    version=version,
    author="basedet team",
    python_requires=">=3.6",
    long_description=long_description,
    ext_modules=get_extensions(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    packages=setuptools.find_packages(),
)
