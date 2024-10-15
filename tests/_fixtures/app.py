__all__ = ["apptest", "mock_openai", "mock_chat_input", "mock_agent_stream", "mock_text_file"]

import pytest
from unittest.mock import MagicMock, ANY, patch
import os
import typing as ty
from loguru import logger
from io import BytesIO

import streamlit as st
from streamlit.testing.v1 import AppTest
from openai import OpenAI
from langchain_core.messages import AIMessage
from langgraph.graph.graph import CompiledGraph


import svsvllm.__main__ as main


@pytest.fixture
def apptest(trace_logging_level: bool) -> AppTest:
    """App for testing."""
    path = os.path.abspath(main.__file__)
    logger.debug(f"Loading script: {path}")
    at = AppTest.from_file(path, default_timeout=30)
    return at


@pytest.fixture
def mock_openai() -> ty.Iterator[MagicMock]:
    """Mock `OpenAI` client.

    The mock happens "twice":
    * we create a mock that we can inject into the session's state;
    * we mock `OpenAI.__new__`, so that it returns a mock on client creation.
    """
    # Create mock client
    openai_client = MagicMock()
    # Mock method that returns LLM's response
    return_value = MagicMock()
    return_value.response.choices.return_value = [MagicMock()]
    return_value.response.choices[0].message.content = "Hi."
    openai_client.chat.completions.create.return_value = return_value
    # Also mock the class `OpenAI` to return a `MagicMock`
    # And again mock method that returns LLM's response
    with patch.object(OpenAI, "__new__", return_value=MagicMock()) as openaimock:
        openaimock.client.chat.completions.create = return_value
        yield openai_client


@pytest.fixture
def mock_chat_input() -> ty.Iterator[MagicMock]:
    """Mock first few user inputs."""
    with patch.object(st, "chat_input", side_effect=["Hello", None]) as chat_input:
        yield chat_input


@pytest.fixture
def mock_agent_stream() -> ty.Iterator[MagicMock]:
    """Mock agent's `CompiledGraph.stream()` method."""
    return_value = MagicMock()
    return_value["messages"] = [AIMessage(content="This is a mocked message.")]
    with patch.object(CompiledGraph, "stream", side_effect=[return_value]) as mock_stream:
        yield mock_stream


@pytest.fixture
def mock_text_file() -> BytesIO:
    """Mock file content (can be any file type you're expecting)."""
    mock_file = BytesIO(b"Test file content")
    mock_file.name = "test-file.txt"
    return mock_file
