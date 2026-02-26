"""Setup script for MIP Splatting CUDA extension"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA architecture flags
cuda_arch_list = [
    '-gencode', 'arch=compute_75,code=sm_75',  # RTX 20xx, Titan RTX
    '-gencode', 'arch=compute_80,code=sm_80',  # A100
    '-gencode', 'arch=compute_86,code=sm_86',  # RTX 30xx
    '-gencode', 'arch=compute_89,code=sm_89',  # RTX 40xx
]

setup(
    name='mip_splatting_cuda',
    ext_modules=[
        CUDAExtension(
            name='mip_splatting_cuda',
            sources=['mip_splat_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                ] + cuda_arch_list
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
