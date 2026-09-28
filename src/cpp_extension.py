"""
C++ extension loader with automated JIT compilation and seamless Python fallback.
Ensures zero-downtime execution regardless of compiler availability.
"""

import os
import sys
from typing import Optional, Callable
from utils.logger import get_logger

logger = get_logger(__name__)

_CPP_OPS = None
_CPP_LOAD_ATTEMPTED = False


def load_cpp_ops() -> Optional[object]:
    """
    Load or compile the news_fast_ops C++ module.
    Attempts:
      1. Direct import of compiled module (AOT).
      2. JIT compilation using torch.utils.cpp_extension.load.
      3. Fallback to None (which triggers pure Python implementations).
    """
    global _CPP_OPS, _CPP_LOAD_ATTEMPTED
    if _CPP_LOAD_ATTEMPTED:
        return _CPP_OPS

    _CPP_LOAD_ATTEMPTED = True

    # 1. Try importing pre-built AOT module
    try:
        import news_fast_ops
        _CPP_OPS = news_fast_ops
        logger.info("Successfully loaded pre-compiled C++ module: news_fast_ops")
        return _CPP_OPS
    except ImportError:
        pass

    # 2. Try JIT compiling via torch.utils.cpp_extension
    try:
        import torch
        from torch.utils.cpp_extension import load

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        csrc_dir = os.path.join(project_root, "csrc")

        sources = [
            os.path.join(csrc_dir, "fast_collator.cpp"),
            os.path.join(csrc_dir, "fast_metrics.cpp"),
            os.path.join(csrc_dir, "bindings.cpp"),
        ]

        if all(os.path.exists(s) for s in sources):
            logger.info("AOT module not found. Attempting JIT compilation of C++ extensions...")
            extra_cflags = ["/O2", "/openmp", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-fopenmp", "-std=c++17"]
            extra_ldflags = [] if sys.platform == "win32" else ["-fopenmp"]

            _CPP_OPS = load(
                name="news_fast_ops",
                sources=sources,
                extra_cflags=extra_cflags,
                extra_ldflags=extra_ldflags,
                verbose=False,
            )
            logger.info("JIT compilation of news_fast_ops succeeded!")
            return _CPP_OPS
    except Exception as e:
        logger.warning(
            f"C++ acceleration unavailable ({e}). Using optimized pure PyTorch/Python fallback."
        )

    _CPP_OPS = None
    return None


def is_cpp_available() -> bool:
    """Check if C++ accelerated ops are active."""
    return load_cpp_ops() is not None


def get_fast_collator() -> Optional[Callable]:
    """Return C++ fast collator if available, else None."""
    ops = load_cpp_ops()
    return getattr(ops, "fast_left_pad_collate", None) if ops else None


def get_fast_metrics() -> Optional[Callable]:
    """Return C++ fast metrics computation if available, else None."""
    ops = load_cpp_ops()
    return getattr(ops, "fast_compute_metrics", None) if ops else None
