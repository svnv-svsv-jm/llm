import pytest
from unittest.mock import patch, Mock, MagicMock
from loguru import logger
import typing as ty
import sys, os

import streamlit as st
from streamlit.testing.v1 import AppTest
from langchain_core.messages import AIMessage
from langchain_huggingface import ChatHuggingFace

from svsvllm.defaults import DEFAULT_LLM
from svsvllm.ui.session_state import SessionState
from svsvllm.utils import CommandTimer
from svsvllm.settings import settings


@pytest.mark.parametrize("openai_api_key", [None, "any"])
def test_ui(
    apptest_ss: AppTest,
    res_docs_path: str,
    mock_openai: MagicMock,
    mock_chat_input: MagicMock,
    mock_agent_stream: MagicMock,
    # mock_hf_model_creation: dict[str, MagicMock],
    # mock_transformers_pipeline: MagicMock,
    # mock_hf_pipeline: MagicMock,
    # mock_hf_chat: MagicMock,
    openai_api_key: str | None,
) -> None:
    """Test app is ready to work with or without OpenAI model.

    Args:
        openai_api_key (str | None):
            When `None`, the open source model will be used.
    """
    apptest = apptest_ss

    # Inject key
    SessionState().state["openai_api_key"] = openai_api_key
    apptest.session_state.openai_api_key = openai_api_key

    # Run app with mocked user inputs
    with patch.object(settings, "uploaded_files_dir", res_docs_path):
        # NOTE: we may even `apptest.chat_input[0].set_value("Hi").run()` but we have to run first once
        with CommandTimer("apptest.run") as timer:
            apptest.run()
        logger.info(f"App run took {timer.elapsed_time} seconds.")

    # Test: OpenAI key
    if openai_api_key is not None:
        assert SessionState().state.openai_api_key is not None
        assert apptest.session_state.openai_api_key is not None

    # Test HF model name exits regardless
    assert SessionState().state.model_name == DEFAULT_LLM

    # Test no erros were raised
    for i, ex in enumerate(apptest.exception):
        logger.info(f"({i}): {ex}")
    assert len(apptest.exception) == 0

    # Test called
    mock_chat_input.assert_called()  # Input provided

    # If no OpenAI key, check open-source LLM was used
    if openai_api_key is None:
        # mock_transformers_pipeline.assert_called()
        # mock_hf_pipeline.assert_called()
        # mock_hf_chat.assert_called()
        mock_agent_stream.assert_called()
        assert isinstance(apptest.session_state.chat_model, ChatHuggingFace)
    else:
        mock_openai.assert_called()  # OpenAI model created

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

    # End
    logger.success("PASSED")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
