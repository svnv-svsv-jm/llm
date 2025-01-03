import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

from svsvchat.settings import Settings
from svsvchat.session_state import SessionState
from svsvchat.chat import chat_with_user


@pytest.mark.parametrize("system_prompt", [None, "Your name is Antonio"])
def test_chat_with_user(
    session_state: SessionState,
    settings: Settings,
    system_prompt: str,
) -> None:
    """Test `chat_with_user`."""
    with patch.object(
        settings,
        "system_prompt",
        system_prompt,
    ), patch.object(
        settings,
        "prompt_role",
        "user",
    ):
        message = chat_with_user("What is your name?")
    logger.info(message)
    assert message.content

    if system_prompt:
        assert "antonio" in message.content.lower()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
