__all__ = ["log_level_key", "trace_logging_level", "log_file", "log_level", "log_to_file"]

import pytest
import os, sys
from loguru import logger
from pathlib import Path
from types import TracebackType

from svsvchat.const import LOG_LEVEL_KEY
from svsvllm.utils.logger import DEFAULT_FORMAT


@pytest.fixture
def log_level_key() -> str:
    """Logging level key."""
    return f"{LOG_LEVEL_KEY}"


@pytest.fixture
def trace_logging_level(log_level_key: str) -> bool:
    """Sets the logging level to `TRACE`."""
    os.environ[log_level_key] = "TRACE"
    return True


@pytest.fixture
def log_file(artifacts_location: str) -> str:
    """Logging filename."""
    f = os.path.join(artifacts_location, "tests.log")
    if Path(f).exists():
        Path(f).unlink()
    return f


@pytest.fixture
def log_level() -> str | int:
    """Logging level."""
    return "TRACE"


def log_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    """A custom exception handler that logs uncaught exceptions.

    Parameters:
        exc_type (Type[BaseException]):
            The class of the exception.

        exc_value (BaseException):
            The instance of the exception raised.

        exc_traceback (Optional[TracebackType]):
            The traceback object with the stack trace.
    """
    # Log the exception with its traceback
    logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


@pytest.fixture
def log_to_file(log_file: str, log_level: str) -> int:
    """Logs to file."""
    # Set the global exception handler
    __excepthook__ = sys.excepthook

    def _log_(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        log_exception(exc_type, exc_value, exc_traceback)
        __excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _log_

    # Add logging sink
    log_handler_id = logger.add(
        log_file,
        level=log_level,
        format=DEFAULT_FORMAT,
        colorize=False,
    )

    # Return the logger ID
    return log_handler_id
