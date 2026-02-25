"""
Build the NeuroSGM CUDA extension:
    pip install -e .
    or
    python setup.py build_ext --inplace
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0")

setup(
    name="neuro3dgs_cuda",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="neuro3dgs_cuda",
            sources=["mip_splat_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-Xptxas=-v",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
