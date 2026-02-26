"""
Setup file for Gaussian Field CUDA extension

Installation:
    pip install -e . 
    
Or from the end_to_end directory:
    python setup_gaussian_field.py install
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0")

setup(
    name="gaussian_field_cuda",
    version="1.0.0",
    description="CUDA-accelerated Gaussian Field evaluation for 3D implicit functions",
    ext_modules=[
        CUDAExtension(
            name="gaussian_field_cuda",
            sources=["gaussian_field_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-Xptxas=-v",
                    "--fmad=true",  # Fuse multiply-add
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
