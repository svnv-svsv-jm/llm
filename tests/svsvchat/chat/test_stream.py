import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

from svsvchat.session_state import SessionState
from svsvchat.chat import stream_open_source_model


def test_stream_open_source_model(session_state: SessionState, query: str) -> None:
    """Test `stream_open_source_model`."""
    for msg in stream_open_source_model(query):
        logger.info(msg)
        assert msg


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
