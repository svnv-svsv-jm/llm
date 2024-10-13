import pytest
from unittest.mock import patch
from loguru import logger
import typing as ty
import sys

import torch

from svsvllm.utils import pick_single_gpu


def test_pick_single_gpu_on_mac() -> None:
    """Test `pick_single_gpu` raises an error on Mac."""
    if torch.cuda.device_count() == 0:
        with pytest.raises(RuntimeError):
            pick_single_gpu()


def test_pick_single_gpu_on_cuda() -> None:
    """Test `pick_single_gpu` can run on machines with CUDA."""
    if torch.cuda.device_count() > 0:
        gpu_id = pick_single_gpu()
        assert isinstance(gpu_id, int)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
