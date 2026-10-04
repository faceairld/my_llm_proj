#!/usr/bin/env python3
"""
Python CUDA-to-MUSA conversion script for LMCache.
Converts torch.cuda references to torch.musa equivalents.
Does NOT modify internal package import paths (e.g., from .cuda._wrapper).
"""
import os
import re
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Files/directories to skip
SKIP_DIRS = {"__pycache__", ".git", "csrc", "csrc_musa", "csrc_hip", "build", "dist", ".eggs", "*.egg-info", "rust"}
SKIP_FILES = {"convert_python_cuda_to_musa.py", "simple_porting_lmcache.py"}

# Replacement rules - order matters (longer/more specific patterns first)
REPLACEMENTS = [
    # Import torch_musa - add at top of files that use torch.cuda
    # (handled separately below)

    # torch.cuda API calls -> torch.musa
    (r'torch\.cuda\.is_available\(\)', 'torch.musa.is_available()'),
    (r'torch\.cuda\.device_count\(\)', 'torch.musa.device_count()'),
    (r'torch\.cuda\.set_device\(', 'torch.musa.set_device('),
    (r'torch\.cuda\.synchronize\(', 'torch.musa.synchronize('),
    (r'torch\.cuda\.empty_cache\(', 'torch.musa.empty_cache('),
    (r'torch\.cuda\.max_memory_allocated\(', 'torch.musa.max_memory_allocated('),
    (r'torch\.cuda\.max_memory_reserved\(', 'torch.musa.max_memory_reserved('),
    (r'torch\.cuda\.memory_allocated\(', 'torch.musa.memory_allocated('),
    (r'torch\.cuda\.memory_reserved\(', 'torch.musa.memory_reserved('),
    (r'torch\.cuda\.reset_peak_memory_stats\(', 'torch.musa.reset_peak_memory_stats('),
    (r'torch\.cuda\.current_device\(\)', 'torch.musa.current_device()'),
    (r'torch\.cuda\.current_stream\(', 'torch.musa.current_stream('),
    (r'torch\.cuda\.Stream\(', 'torch.musa.Stream('),
    (r'torch\.cuda\.Event\(', 'torch.musa.Event('),
    (r'torch\.cuda\.FloatTensor', 'torch.musa.FloatTensor'),
    (r'torch\.cuda\.HalfTensor', 'torch.musa.HalfTensor'),
    (r'torch\.cuda\.BFloat16Tensor', 'torch.musa.BFloat16Tensor'),
    (r'torch\.cuda\.DoubleTensor', 'torch.musa.DoubleTensor'),
    (r'torch\.cuda\.IntTensor', 'torch.musa.IntTensor'),
    (r'torch\.cuda\.LongTensor', 'torch.musa.LongTensor'),

    # Generic torch.cuda.xxx -> torch.musa.xxx (catch remaining APIs)
    # But NOT torch.cuda._something (internal paths) or torch.cuda. at end of line in imports
    (r'torch\.cuda\.(?!_)(\w+)', r'torch.musa.\1'),

    # import torch.cuda -> import torch_musa (but keep original import for compatibility)
    (r'^import torch\.cuda$', 'import torch_musa'),
    (r'^import torch\.cuda\b', 'import torch_musa'),

    # Device string patterns - be careful with context
    # f"cuda:{...}" -> f"musa:{...}"
    (r'f"cuda:\{', 'f"musa:{'),
    (r"f'cuda:\{", "f'musa:{"),

    # "cuda:0", "cuda:1", etc.
    (r'"cuda:(\d+)"', r'"musa:\1"'),
    (r"'cuda:(\d+)'", r"'musa:\1'"),

    # "cuda" standalone device strings (but not as part of larger words)
    (r'"cuda"', '"musa"'),
    (r"'cuda'", "'musa'"),

    # .cuda() method calls on tensors
    (r'\.cuda\(\)', '.musa()'),

    # .is_cuda attribute
    (r'\.is_cuda\b', '.is_musa'),

    # nccl -> mccl (DDP backend)
    (r'"nccl"', '"mccl"'),
    (r"'nccl'", "'mccl'"),
    (r'backend="nccl"', 'backend="mccl"'),
    (r"backend='nccl'", "backend='mccl'"),
]


def should_skip_file(filepath):
    """Check if file should be skipped."""
    basename = os.path.basename(filepath)
    if basename in SKIP_FILES:
        return True
    parts = filepath.split(os.sep)
    for part in parts:
        if part in SKIP_DIRS:
            return True
        for skip in SKIP_DIRS:
            if '*' in skip and part.endswith(skip.replace('*', '')):
                return True
    return False


def needs_torch_musa_import(content):
    """Check if file uses torch.cuda and needs import torch_musa."""
    return bool(re.search(r'torch\.(cuda|musa)', content))


def add_torch_musa_import(content):
    """Add import torch_musa after existing torch imports."""
    if 'import torch_musa' in content:
        return content

    # Find the best place to insert
    lines = content.split('\n')
    insert_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Find last torch import
        if stripped.startswith('import torch') or stripped.startswith('from torch'):
            insert_idx = i + 1

    if insert_idx == 0:
        # No torch imports found, add after other imports
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                insert_idx = i + 1

    if insert_idx > 0:
        lines.insert(insert_idx, 'import torch_musa')
        return '\n'.join(lines)

    return content


def convert_file(filepath):
    """Convert a single Python file from CUDA to MUSA references."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, FileNotFoundError):
        return 0

    original = content
    count = 0

    for pattern, replacement in REPLACEMENTS:
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        if new_content != content:
            # Count actual replacements
            matches = re.findall(pattern, content, flags=re.MULTILINE)
            count += len(matches)
            content = new_content

    # Add torch_musa import if needed
    if content != original and needs_torch_musa_import(content):
        content = add_torch_musa_import(content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return count

    return 0


def main():
    total_files = 0
    total_replacements = 0
    modified_files = []

    for root, dirs, files in os.walk(PROJECT_DIR):
        # Skip directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith('.egg-info')]

        for fname in files:
            if not fname.endswith('.py'):
                continue

            filepath = os.path.join(root, fname)
            relpath = os.path.relpath(filepath, PROJECT_DIR)

            if should_skip_file(relpath):
                continue

            count = convert_file(filepath)
            if count > 0:
                total_files += 1
                total_replacements += count
                modified_files.append((relpath, count))
                print(f"  {relpath}: {count} replacements")

    print(f"\nTotal: {total_files} files modified, {total_replacements} replacements")
    return modified_files


if __name__ == '__main__':
    modified = main()
