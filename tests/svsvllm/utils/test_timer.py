import pytest
from loguru import logger
import typing as ty
import sys
import time

from svsvllm.utils import CommandTimer


@pytest.mark.parametrize("sleep_time", [1, 6])
def test_command_timer(sleep_time: float) -> None:
    """Test this runs smoothly."""
    timer = CommandTimer()
    timer.run(lambda: time.sleep(sleep_time))
    secs = timer.elapsed_time
    logger.info(f"Elapsed seconds: {secs}")
    assert secs > sleep_time


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
