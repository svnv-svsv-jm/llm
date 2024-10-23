import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

import streamlit as st
from streamlit.testing.v1 import AppTest
from langchain_core.messages import AIMessage
from langchain_huggingface import ChatHuggingFace

from svsvllm.defaults import DEFAULT_LLM
from svsvllm.app.ui.session_state import SessionState
from svsvllm.app.settings import settings
from svsvllm.app.ui import ui
from svsvllm.utils import pretty_format_dict_str

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
    logger.info(f"\n{pretty_format_dict_str(apptest.session_state, indent=2)}")
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


@pytest.mark.parametrize("openai_api_key", [None, "any"])
@pytest.mark.parametrize("state_for_client", [False, True])
def test_openai(
    safe_apptest_ss: AppTest,
    docs_path: str,
    mock_openai: MagicMock,
    mock_chat_input: MagicMock,
    mock_agent_stream: MagicMock,
    # mock_hf_model_creation: dict[str, MagicMock],
    # mock_transformers_pipeline: MagicMock,
    # mock_hf_pipeline: MagicMock,
    # mock_hf_chat: MagicMock,
    state_for_client: bool,
    openai_api_key: str | None,
) -> None:
    """Test app is ready to work with or without OpenAI model.

    Args:
        state_for_client (bool):
            If `True`, we inject the mock client into the state. If `False`, we create a new `OpenAI` client, but still the `OpenAI.__new__` method is mocked.

        openai_api_key (str | None):
            When `None`, the open source model will be used.
    """
    apptest = safe_apptest_ss
    # Inject mock into app state to avoid creation of OpenAI client
    if state_for_client:
        apptest.session_state["openai_client"] = mock_openai

    # Inject key
    apptest.session_state["openai_api_key"] = openai_api_key
    # Inject fake model name
    apptest.session_state["model_name"] = DEFAULT_LLM

    # Run app with mocked user inputs
    with patch.object(settings, "uploaded_files_dir", docs_path):
        apptest.run(timeout=120)

    # Test no erros were raised
    for i, ex in enumerate(apptest.exception):
        logger.info(f"({i}): {ex}")
    assert len(apptest.exception) == 0

    # Test input was provided
    mock_chat_input.assert_called()

    # If no OpenAI key, check open-source LLM was used
    if openai_api_key is None:
        # mock_transformers_pipeline.assert_called()
        # mock_hf_pipeline.assert_called()
        # mock_hf_chat.assert_called()
        mock_agent_stream.assert_called()
        assert isinstance(apptest.session_state.chat_model, ChatHuggingFace)

    # Test: OpenAI key
    assert apptest.session_state.openai_api_key == openai_api_key

    # Test: chat history exists
    logger.info(apptest.session_state.messages)
    assert len(apptest.session_state.messages) == 3

    # Test AIMessage(s) is/are there
    # We are expecting the welcome message, then there is the fake user input, then the mocked response
    counter = 0
    for msg in apptest.session_state.messages:
        if isinstance(msg, AIMessage):
            counter += 1
    assert counter == 2


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
