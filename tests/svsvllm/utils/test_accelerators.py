import pytest
from unittest.mock import patch
from loguru import logger
import typing as ty
import sys

import torch
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator

from svsvllm.utils import choose_auto_accelerator, find_device


def test_find_device() -> None:
    """Test `find_device` for coverage."""
    device = find_device()
    assert isinstance(device, torch.device)


def test_choose_auto_accelerator_mps() -> None:
    """Test `choose_auto_accelerator` on MPS device."""
    with patch.object(MPSAccelerator, "is_available", return_value=True):
        assert choose_auto_accelerator() == "mps"


def test_choose_auto_accelerator_cuda() -> None:
    """Test `choose_auto_accelerator` on CUDA device."""
    with patch.object(MPSAccelerator, "is_available", return_value=False):
        with patch.object(CUDAAccelerator, "is_available", return_value=True):
            assert "cuda" in choose_auto_accelerator()


def test_choose_auto_accelerator_cpu() -> None:
    """Test `choose_auto_accelerator` on CPU-only device."""
    with patch.object(MPSAccelerator, "is_available", return_value=False):
        with patch.object(CUDAAccelerator, "is_available", return_value=False):
            assert "cpu" in choose_auto_accelerator()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
