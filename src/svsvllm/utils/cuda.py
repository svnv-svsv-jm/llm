__all__ = ["pick_single_gpu", "find_usable_cuda_devices"]

import typing as ty

import torch
from pytorch_lightning.accelerators import find_usable_cuda_devices

from svsvllm.exceptions import NoGPUError


def pick_single_gpu(exclude_gpus: ty.List[int] = []) -> int:  # pragma: no cover
    """
    Raises:
        RuntimeError:
            If you try to allocate a GPU, when no GPUs are available.
    """
    # Initialize
    previously_used_gpus = []
    unused_gpus = []
    # Iterate over devices
    for i in range(torch.cuda.device_count()):
        if i in exclude_gpus:
            continue
        if torch.cuda.memory_reserved(f"cuda:{i}") > 0:
            previously_used_gpus.append(i)
        else:
            unused_gpus.append(i)
    # Prioritize previously used GPUs
    for i in previously_used_gpus + unused_gpus:
        # Try to allocate on device:
        device = torch.device(f"cuda:{i}")
        try:
            torch.ones(1).to(device)
        except Exception:
            continue
        return i
    # Raise error if no GPUs
    raise NoGPUError("No GPUs available.")
