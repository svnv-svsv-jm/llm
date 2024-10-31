import pytest
import typing as ty
from loguru import logger

import torch

from svsvllm.utils import find_device


@pytest.fixture
def device(request: pytest.FixtureRequest) -> ty.Iterator[torch.device]:
    """Torch device."""
    name = getattr(request, "param", "auto")
    if name is None:
        name = "auto"
    device = find_device(name)
    logger.debug(f"Device: {device}")
    # torch.set_default_device(device)
    with torch.device(device):
        yield device
