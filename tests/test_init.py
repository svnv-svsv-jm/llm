import pytest
from loguru import logger
import typing as ty
import sys


def test_placeholder() -> None:
    """To be removed when real tests are implemented."""


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
