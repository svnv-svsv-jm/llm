import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys

import torch

from svsvllm.utils import pick_single_gpu
from svsvllm.exceptions import NoGPUError


def test_pick_single_gpu_on_mac() -> None:
    """Test `pick_single_gpu` raises an error on Mac."""
    with patch.object(torch.cuda, "device_count", return_value=0):
        assert torch.cuda.device_count() == 0
        with pytest.raises(NoGPUError):
            pick_single_gpu()


def test_pick_single_gpu_on_cuda() -> None:
    """Test `pick_single_gpu` can run on machines with CUDA."""
    with patch.object(torch.cuda, "device_count", return_value=1):
        assert torch.cuda.device_count() == 1
        try:
            gpu_id = pick_single_gpu()
            assert isinstance(gpu_id, int)
        except NoGPUError as ex:
            logger.info(ex)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
