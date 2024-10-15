__all__ = ["find_device", "choose_auto_accelerator"]

from loguru import logger
import torch
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator

from .cuda import pick_single_gpu


def find_device(accelerator: str = "auto") -> torch.device:
    """Automatically finds system's device for PyTorch."""
    device = choose_auto_accelerator(accelerator)
    return torch.device(device)


@logger.catch(Exception, default="cpu")
def choose_auto_accelerator(accelerator_flag: str = "auto") -> str:
    """Choose the accelerator type (str) based on availability when `accelerator='auto'`.

    Returns:
        (str): Either `mps`, `cuda`, `cuda:{i}` or `cpu`.
    """
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
