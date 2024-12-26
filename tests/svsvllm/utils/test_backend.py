import pytest
from unittest.mock import patch
from loguru import logger
import typing as ty
import sys

import torch
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator

from svsvllm.utils import get_default_backend


def test_choose_auto_accelerator_none() -> None:
    """Test `get_default_backend`."""
    with patch.object(MPSAccelerator, "is_available", return_value=False):
        with patch.object(CUDAAccelerator, "is_available", return_value=False):
            assert "x86" == get_default_backend()


def test_choose_auto_accelerator_cuda() -> None:
    """Test `get_default_backend`."""
    with patch.object(MPSAccelerator, "is_available", return_value=False):
        with patch.object(CUDAAccelerator, "is_available", return_value=True):
            assert "fbgemm" == get_default_backend()


def test_choose_auto_accelerator_mps() -> None:
    """Test `get_default_backend`."""
    with patch.object(MPSAccelerator, "is_available", return_value=True):
        with patch.object(CUDAAccelerator, "is_available", return_value=False):
            assert "qnnpack" == get_default_backend()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
