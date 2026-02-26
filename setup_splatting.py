"""
Setup script for Gaussian Splatting CUDA extension.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

setup(
    name='gaussian_splatting_cuda',
    ext_modules=[
        CUDAExtension(
            name='gaussian_splatting_cuda',
            sources=['gaussian_splatting_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
