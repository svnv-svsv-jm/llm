import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest
from langchain_core.messages import AIMessage

from svsvllm.defaults import DEFAULT_LLM


def test_sidebar(
    apptest: AppTest,
    mock_openai: MagicMock,
    mock_chat_input: MagicMock,
    mock_agent_stream: MagicMock,
) -> None:
    """Test sidebar's settings."""
    apptest.run()
    assert not apptest.exception


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
