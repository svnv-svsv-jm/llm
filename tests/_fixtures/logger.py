__all__ = ["log_level_key", "trace_logging_level"]

import pytest
import os

from svsvchat.const import LOG_LEVEL_KEY


@pytest.fixture
def log_level_key() -> str:
    """Logging level key."""
    return f"{LOG_LEVEL_KEY}"


@pytest.fixture
def trace_logging_level(log_level_key: str) -> bool:
    """Sets the logging level to `TRACE`."""
    os.environ[log_level_key] = "TRACE"
    return True
