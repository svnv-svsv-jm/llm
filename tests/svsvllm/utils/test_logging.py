import pytest
import sys
import os
import typing as ty
from loguru import logger

from svsvllm.utils import set_up_logging


def _log(msg: str = "Hello") -> None:  # pragma: no cover
    """Only needed for testing."""
    logger.info(msg)


@pytest.mark.parametrize(
    "level, serialize, show_file_info",
    [
        ("INFO", True, True),
        ("DEBUG", False, True),
        ("TRACE", False, False),
    ],
)
def test_logging(level: str, serialize: bool, show_file_info: bool) -> None:
    """Test logging module."""
    logger_id = set_up_logging(level=level, serialize=serialize, show_file_info=show_file_info)
    logger.info("Testing...")
    with logger.contextualize(key="value", metoo="ok"):
        logger.info("Hello with extras!")
        logger.info("Hello with extras!")
    _log("Hello")
    logger.remove(logger_id)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
