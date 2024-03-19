import pytest
from loguru import logger
import typing as ty
import sys

from svsvllm.utils.nb import nb_init


def test_nb() -> None:
    """To be removed when real tests are implemented."""
    nb_init()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
