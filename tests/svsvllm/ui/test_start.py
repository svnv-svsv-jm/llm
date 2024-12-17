import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

import streamlit as st
from streamlit.testing.v1 import AppTest

from svsvllm.ui.session_state import SessionState
from svsvllm.settings import settings

# TODO: see https://medium.com/@chrisschneider/build-a-high-quality-streamlit-app-with-test-driven-development-eef4e462f65e


@patch.object(settings, "has_chat", False)
@patch.object(settings, "has_sidebar", False)
def test_app_starts(safe_apptest_ss: AppTest) -> None:
    """Verify the app starts without errors."""
    apptest = safe_apptest_ss
    apptest.run(timeout=3)
    for i, ex in enumerate(apptest.exception):
        logger.info(f"({i}): {ex}")
    assert len(apptest.exception) == 0
    assert apptest.session_state["chat_activated"] == False
    assert "_bind" not in apptest.session_state


@patch.object(settings, "has_chat", False)
@patch.object(settings, "has_sidebar", False)
def test_page_title(safe_apptest_ss: AppTest) -> None:
    """Verify the app has the expected title."""
    apptest = safe_apptest_ss
    with patch.object(st, "title") as mock_title, patch.object(
        st,
        "title",
    ) as mock_title, patch.object(
        st,
        "subheader",
    ) as mock_subheader, patch.object(
        st,
        "caption",
    ) as mock_caption:
        apptest.run(timeout=3)
    # Test they were was called once with the correct argument
    mock_title.assert_called_once_with(settings.app_title)
    mock_subheader.assert_called_once_with(settings.app_subheader)
    mock_caption.assert_called_once_with(settings.app_caption)


@patch.object(settings, "has_chat", False)
@patch.object(settings, "has_sidebar", False)
def test_shows_welcome_message(safe_apptest_ss: AppTest) -> None:
    """Verify initial message from assistant is shown."""
    apptest = safe_apptest_ss
    with patch.object(st, "write") as mock_msg:
        apptest.run(timeout=2)
    # Test there is only one message
    assert len(apptest.session_state["messages"]) == 1
    # Get value of message and assert called with it
    msg = SessionState().state.messages[0]
    assert msg.content == settings.start_message_en
    mock_msg.assert_called_once_with(msg.content)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
