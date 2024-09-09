__all__ = ["get_default_backend"]

from loguru import logger
import torch
from .accelerators import choose_auto_accelerator


def get_default_backend(raise_error_if_engine_not_found: bool = False) -> str:
    """Returns the default backend. Backend is a string representing the target backend.
    Currently supports `x86`, `fbgemm`, `qnnpack` and `onednn`."""
    # Get device
    acc = choose_auto_accelerator("auto")

    # Engines
    engine = "x86"
    engines = torch.backends.quantized.supported_engines

    # Cuda
    if "cuda" in acc:
        engine = "fbgemm"

    # MPS
    if "mps" in acc:
        engine = "qnnpack"

    # Check engine exists
    if engine not in engines:
        msg = f"Engine `{engine}` not available: {engines}"
        if raise_error_if_engine_not_found:
            raise RuntimeError(msg)
        logger.warning(msg)

    # Return engine
    return engine
