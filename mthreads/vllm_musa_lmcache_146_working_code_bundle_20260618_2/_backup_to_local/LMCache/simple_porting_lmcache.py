#!/usr/bin/env python3
"""
SimplePorting script for LMCache CUDA-to-MUSA migration.
Run inside the container:
    python3 simple_porting_lmcache.py
"""
import os
import shutil
from torch_musa.utils.simple_porting import SimplePorting

PROJECT_DIR = "/workspace/LMCache"

# Custom mapping rules (project-specific, extend defaults)
MAPPING_RULES = {
    # torch_musa header includes (quoted includes, NOT angle brackets)
    '#include <c10/cuda/CUDAGuard.h>': '#include "torch_musa/csrc/core/MUSAGuard.h"',
    '#include <ATen/cuda/CUDAContext.h>': '#include "torch_musa/csrc/aten/musa/MUSAContext.h"',

    # Namespace fixes
    "at::cuda::": "at::musa::",
    "c10::cuda::": "c10::musa::",

    # Type names
    "OptionalCUDAGuard": "OptionalMUSAGuard",

    # Device type checks
    ".is_cuda()": ".is_privateuseone()",

    # Compiler macros
    "__NVCC__": "__MUSACC__",

    # Local headers (.cuh -> .muh)
    '"mem_kernels.cuh"': '"mem_kernels.muh"',
    '"cachegen_kernels.cuh"': '"cachegen_kernels.muh"',
    '"pos_kernels.cuh"': '"pos_kernels.muh"',
    '"cuda_compat.h"': '"cuda_compat.h"',
    '"dispatch_utils.h"': '"dispatch_utils.h"',

    # USE_ROCM guard for MUSA - use USE_MUSA instead
    "#ifdef USE_ROCM": "#if defined(USE_ROCM) || defined(USE_MUSA)",
    # cuda_fp8.h -> musa_fp8.h
    "#include <cuda_fp8.h>": "#include <musa_fp8.h>",

    # CHECK_CUDA_CALL macro name (just update the string references)
    'fprintf(stderr, "CUDA error in file': 'fprintf(stderr, "MUSA error in file',
    '"CUDA error in file \'")': '"MUSA error in file \'"))',
}


def clean_existing_musa_dirs():
    """Remove existing _musa directories before regeneration."""
    path = os.path.join(PROJECT_DIR, "csrc_musa")
    if os.path.exists(path):
        print(f"Removing: {path}")
        shutil.rmtree(path)


def port_directories():
    """Port each CUDA directory using SimplePorting."""
    dirs_to_port = [
        # (cuda_dir, ignore_dirs)
        (os.path.join(PROJECT_DIR, "csrc"), []),
    ]
    for cuda_dir, ignore_dirs in dirs_to_port:
        if os.path.exists(cuda_dir):
            print(f"Porting {cuda_dir}")
            SimplePorting(
                cuda_dir_path=cuda_dir,
                ignore_dir_paths=ignore_dirs,
                mapping_rule=MAPPING_RULES,
                drop_default_mapping=False,
            ).run()


if __name__ == "__main__":
    clean_existing_musa_dirs()
    port_directories()
    print("SimplePorting complete.")
