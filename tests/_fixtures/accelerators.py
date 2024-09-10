import pytest
from loguru import logger

import torch

from svsvllm.utils import find_device


@pytest.fixture
def device() -> torch.device:
    """Torch device."""
    device = find_device()
    logger.debug(f"Device: {device}")
    return device
