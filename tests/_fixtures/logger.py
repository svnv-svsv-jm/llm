import pytest
import os

from svsvllm.app.const import LOG_LEVEL_KEY


@pytest.fixture
def log_level_key() -> str:
    """Logging level key."""
    return LOG_LEVEL_KEY


@pytest.fixture
def trace_logging_level(log_level_key: str) -> bool:
    """Sets the logging level to `TRACE`."""
    os.environ[log_level_key] = "TRACE"
    return True
