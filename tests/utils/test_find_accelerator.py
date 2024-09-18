import pytest
from unittest.mock import patch
from loguru import logger
import typing as ty
import sys

from svsvllm.utils import find_accelerator


def test_find_accelerator_on_darwin() -> None:
    """Test `find_accelerator` returns `"cpu"` if on Mac."""
    with patch.object(sys, "platform", new="darwin"):
        device = find_accelerator()
        assert device.lower() == "cpu"


def test_find_accelerator_on_else() -> None:
    """Test `find_accelerator` returns `"auto"` if not on Mac."""
    with patch.object(sys, "platform", new="nothing"):
        device = find_accelerator()
        assert device.lower() == "auto"


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
