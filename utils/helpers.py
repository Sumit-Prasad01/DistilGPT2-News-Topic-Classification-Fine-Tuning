"""
Helper utilities for DistilGPT2 News Topic Classification.
Includes device detection, reproducible random seed configuration, execution timers, and system metrics.
"""

import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


def get_device() -> Any:
    """
    Detect and return the best available PyTorch compute device.
    Prioritizes CUDA -> Apple Silicon (MPS) -> CPU.
    """
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Please install torch before requesting device.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across random, numpy, and torch for strict reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string (e.g. '1h 24m 12.4s' or '3.52s').
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


class Timer:
    """
    Context manager and utility to measure code execution duration.

    Usage:
        with Timer("Data Tokenization"):
            tokenize(...)
    """

    def __init__(self, description: str = "Operation", logger=None):
        self.description = description
        self.logger = logger
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        msg = f"[{self.description}] Completed in {format_duration(self.elapsed)}"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


def get_system_info() -> Dict[str, Any]:
    """
    Gather runtime system and hardware information.
    """
    info = {
        "os": sys.platform,
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }

    if torch is not None:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["device_name"] = "Apple Silicon MPS"
        else:
            info["device_name"] = "CPU"
    else:
        info["torch_version"] = "Not Installed"
        info["device_name"] = "Unknown"

    return info
