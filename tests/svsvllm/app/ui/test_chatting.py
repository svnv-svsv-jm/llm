import pytest
from unittest.mock import patch, Mock, MagicMock
from loguru import logger
import typing as ty
import sys, os

import streamlit as st
from streamlit.testing.v1 import AppTest
from langchain_core.messages import AIMessage

from svsvllm.defaults import DEFAULT_LLM
from svsvllm.app.ui.session_state import SessionState
from svsvllm.utils import CommandTimer


def test_chatting(
    apptest_ss: AppTest,
    mock_rag_docs: str,
    mock_chat_input: MagicMock,
    mock_agent_stream: MagicMock,
) -> None:
    """Test chatting."""
    apptest = apptest_ss

    # Inject key
    SessionState().state["openai_api_key"] = None
    apptest.session_state.openai_api_key = None

    # NOTE: we may even `apptest.chat_input[0].set_value("Hi").run()` but we have to run once first
    with CommandTimer("apptest.run") as timer:
        apptest.run(timeout=300)
    logger.info(f"App run took {timer.elapsed_time} seconds.")

    # Test HF model name exits regardless
    assert SessionState().state.model_name == DEFAULT_LLM

    # Test no erros were raised
    for i, ex in enumerate(apptest.exception):
        stack_trace = "\n".join(ex.stack_trace)
        logger.error(f"({i}): {ex.message}\n{stack_trace}")
        error = Exception(ex.message)
        raise error
    assert len(apptest.exception) == 0

    # Test called
    mock_chat_input.assert_called()  # Input provided
    mock_agent_stream.assert_called()  # Mocked response

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
