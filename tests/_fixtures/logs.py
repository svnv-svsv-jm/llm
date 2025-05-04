import pytest
import typing as ty
from loguru import logger
import json

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
