import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

from svsvchat.session_state import SessionState
from svsvchat.chat import chat_with_user


def test_chat_with_user(session_state: SessionState, query: str) -> None:
    """Test `chat_with_user`."""
    message = chat_with_user(query)
    logger.info(message)
    assert message.content


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
