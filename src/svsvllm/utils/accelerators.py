__all__ = ["find_accelerator", "find_device", "choose_auto_accelerator"]

import sys
import torch
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator

from .cuda import pick_single_gpu


def find_accelerator() -> str:
    """Finds the accelerator. Will return `"cpu"` if on Mac, else it will return `"auto"`."""
    if sys.platform.lower() in ["darwin"]:
        return "cpu"
    return "auto"


def find_device(accelerator: str = "auto") -> torch.device:
    """Automatically finds system's device for PyTorch."""
    device = choose_auto_accelerator(accelerator)
    return torch.device(device)


def choose_auto_accelerator(accelerator_flag: str = "auto") -> str:
    """Choose the accelerator type (str) based on availability when `accelerator='auto'`.

    Returns:
        (str): Either `mps`, `cuda`, `cuda:{i}` or `cpu`.
    """
    try:
        if accelerator_flag.lower() == "auto":
            if MPSAccelerator.is_available():
                return "mps"
            if CUDAAccelerator.is_available():
                try:
                    i = pick_single_gpu()
                    return f"cuda:{i}"
                except Exception:
                    return "cuda"
        return "cpu"
    except Exception:
        return "cpu"
