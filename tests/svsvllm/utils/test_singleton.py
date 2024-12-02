import pytest
from loguru import logger
import typing as ty
import sys

from svsvllm.utils import Singleton


class ForTesting(metaclass=Singleton):
    """Does nothing."""


def test_singleton() -> None:
    """Test `Singleton`."""
    assert ForTesting() is ForTesting()
    assert ForTesting._instances
    assert ForTesting._instances.get(ForTesting) is ForTesting()  # type: ignore
    ForTesting.reset(ForTesting)
    assert ForTesting._instances.get(ForTesting) is None  # type: ignore


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
