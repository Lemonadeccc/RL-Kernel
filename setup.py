# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    from torch.utils.cpp_extension import ROCMExtension
except ImportError:
    ROCMExtension = None
import torch


def get_extensions():
    extensions = []

    is_rocm = torch.version.hip is not None

    if is_rocm:
        extensions.append(
            ROCMExtension(
                name="kernel_align_ops",
                sources=[
                    "csrc/ops.cpp",
                    "csrc/fused_logp_kernel.cpp",  # ROCm uses .cpp for HIP kernels
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "hipcc": ["-O3", "--use_fast_math", "-Xhipcc", "-compress-all"],
                },
            )
        )
    elif torch.cuda.is_available():
        extensions.append(
            CUDAExtension(
                name="kernel_align_ops",
                sources=[
                    "csrc/ops.cpp",
                    "csrc/fused_logp_kernel.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": ["-O3", "--use_fast_math", "-Xfatbin", "-compress-all"],
                },
            )
        )
    return extensions


setup(
    name="rl-engine",
    version="0.1.0",
    packages=find_packages(include=["rl_engine", "rl_engine.*"]),
    install_requires=[
        "torch>=2.4.0",
        "tabulate",
        "numpy",
        "accelerate",
        "transformers",
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    extras_require={
        "cuda": ["flashinfer"],
        "rocm": ["aiter"],
    },
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
