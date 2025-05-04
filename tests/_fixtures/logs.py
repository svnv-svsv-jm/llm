import json
import typing as ty

import pytest
from loguru import logger

import svsv


@pytest.fixture
def log_level() -> str:
    """Log level."""
    return "TRACE"


@pytest.fixture(autouse=True)
def enable_logs() -> ty.Iterator[bool]:
    """Enable logs."""
    svsv.configure_logger(True)
    yield True


@pytest.fixture(autouse=True)
def set_up_logging(log_level: str) -> ty.Iterator[int]:
    """Creates logger for tests."""
    yield svsv.set_up_logging(log_level=log_level)


@pytest.fixture
def catch_logs(log_level: str) -> ty.Iterator[list[dict]]:
    """Capture logs."""
    logs: list[dict] = []

    def sink_function(record: str) -> None:
        """Receives serialized records."""
        r: dict = json.loads(record)
        logs.append(r)

    logger_id = logger.add(sink_function, level=log_level, serialize=True)

    yield logs

    logger.remove(logger_id)
