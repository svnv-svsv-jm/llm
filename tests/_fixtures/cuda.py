__all__ = ["device_count", "memory_reserved"]

import pytest
from unittest.mock import patch
import typing as ty
import torch


@pytest.fixture
def device_count(request: pytest.FixtureRequest) -> ty.Iterator[int]:
    """Emulate having GPUs."""
    gpus = getattr(request, "param", 3)
    with patch.object(torch.cuda, "device_count", return_value=gpus):
        yield torch.cuda.device_count()


@pytest.fixture
def memory_reserved(request: pytest.FixtureRequest) -> ty.Iterator[int]:
    """Amount of GPU memory reserved for any GPU."""
    memory = getattr(request, "param", 3)
    with patch.object(torch.cuda, "memory_reserved", return_value=memory):
        yield torch.cuda.memory_reserved(f"cuda:0")
