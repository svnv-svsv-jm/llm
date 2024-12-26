import pytest
from unittest.mock import patch
from loguru import logger
import typing as ty
import sys

import torch

from svsvllm.exceptions import NoGPUError
from svsvllm.utils import pick_single_gpu


@pytest.mark.parametrize("memory_reserved", [0, 1], indirect=True)
@pytest.mark.parametrize("exclude_gpus", [[], [0]])
def test_pick_single_gpu_error(
    exclude_gpus: list[int],
    device_count: int,
    memory_reserved: int,
) -> None:
    """Test `pick_single_gpu` raises the `NoGPUError`."""

    def _raise_error() -> None:
        raise NoGPUError()

    with patch.object(torch.Tensor, "to", side_effect=_raise_error):
        with pytest.raises(NoGPUError):
            pick_single_gpu(exclude_gpus)

    logger.success("ok")


@pytest.mark.parametrize("memory_reserved", [0, 1], indirect=True)
@pytest.mark.parametrize("exclude_gpus", [[], [0]])
def test_pick_single_gpu(
    exclude_gpus: list[int],
    device_count: int,
    memory_reserved: int,
) -> None:
    """Test `pick_single_gpu`."""
    with patch.object(
        torch.Tensor,
        "to",
        return_value=torch.zeros(1),
    ):
        gpu = pick_single_gpu(exclude_gpus)

    assert gpu == 0 if not exclude_gpus else 1

    logger.success("ok")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
