__all__ = ["pick_single_gpu", "find_usable_cuda_devices"]

import typing as ty
from loguru import logger

import torch
from pytorch_lightning.accelerators import find_usable_cuda_devices

from svsvllm.exceptions import NoGPUError


def pick_single_gpu(exclude_gpus: list[int] = []) -> int:
    """
    Raises:
        RuntimeError:
            If you try to allocate a GPU, when no GPUs are available.
    """
    # Initialize
    previously_used_gpus = []
    unused_gpus = []

    # Iterate over devices
    device_count = torch.cuda.device_count()
    logger.trace(f"Device count: {device_count}")
    for i in range(device_count):
        if i in exclude_gpus:
            logger.trace(f"Excluding GPU {i}")
            continue
        mem = torch.cuda.memory_reserved(f"cuda:{i}")
        logger.trace(f"Reserved memory (gpu:{i}): {mem}")
        if mem > 0:
            previously_used_gpus.append(i)
        else:
            unused_gpus.append(i)

    # Prioritize previously used GPUs
    gpus = previously_used_gpus + unused_gpus
    logger.trace(f"Unused GPUs: {gpus}")
    for i in gpus:
        # Try to allocate on device:
        device = torch.device(f"cuda:{i}")
        try:
            torch.ones(1).to(device)
        except Exception:
            continue
        return i

    # Raise error if no GPUs
    raise NoGPUError("No GPUs available.")
