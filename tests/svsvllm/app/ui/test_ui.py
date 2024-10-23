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
from svsvllm.utils import CommandTimer
from svsvllm.app.settings import settings


@pytest.mark.parametrize("openai_api_key", [None, "any"])
@pytest.mark.parametrize("state_for_client", [False, True])
def test_chatting(
    safe_apptest_ss: AppTest,
    res_docs_path: str,
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
    with patch.object(settings, "uploaded_files_dir", res_docs_path):
        with CommandTimer("apptest.run") as timer:
            apptest.run(timeout=120)
        logger.info(f"App run took {timer.elapsed_time} seconds.")

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
