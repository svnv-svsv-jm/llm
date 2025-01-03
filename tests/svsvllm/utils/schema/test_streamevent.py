import pytest
import sys
import os
import typing as ty
from loguru import logger

from svsvllm.schema import ChatMLXEvent, AgentPayload


def test_ChatMLXEvent() -> None:
    """Test `ChatMLXEvent`."""
    event = ChatMLXEvent(payload=AgentPayload())
    assert ChatMLXEvent.is_valid(event.model_dump())
    assert not ChatMLXEvent.is_valid({"timestamp": "ok"})


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
