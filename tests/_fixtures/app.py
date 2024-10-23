__all__ = [
    "apptest",
    "apptest_ss",
    "safe_apptest",
    "safe_apptest_ss",
    "session_state",
    "mock_openai",
    "mock_chat_input",
    "mock_agent_stream",
    "mock_text_file",
]

import pytest
from unittest.mock import MagicMock, patch
import os
import typing as ty
from loguru import logger
from io import BytesIO

from openai import OpenAI
from langchain_core.messages import AIMessage
from langgraph.graph.graph import CompiledGraph
import streamlit as st
from streamlit.testing.v1 import AppTest

import svsvllm.__main__ as main
from svsvllm.app.ui.session_state import SessionState


@pytest.fixture
def session_state() -> ty.Iterator[SessionState]:
    """Session state."""
    with SessionState(reverse=True, auto_sync=False) as ss:
        yield ss


@pytest.fixture
def apptest(trace_logging_level: bool) -> ty.Iterator[AppTest]:
    """App for testing."""
    # Create app from file
    path = os.path.abspath(main.__file__)
    logger.debug(f"Loading script: {path}")
    at = AppTest.from_file(path, default_timeout=30)
    yield at


@pytest.fixture
def apptest_ss(apptest: AppTest, session_state: SessionState) -> ty.Iterator[AppTest]:
    """App for testing with bound session state."""
    session_state.bind(apptest.session_state)
    yield apptest


@pytest.fixture
def safe_apptest(apptest: AppTest) -> ty.Iterator[AppTest]:
    """Patch this method so that we can time out without errors.

    No idea why calling `run()` times out, but we do not care.
    """
    # Keep ref to original method
    __run = apptest.run

    # Patch the run method to time out safely
    def run(**kwags: ty.Any) -> AppTest | None:
        """Wrap original method in a `try-except` statement."""
        logger.debug("Running patched `run`")
        try:
            return __run(**kwags)
        except RuntimeError as e:
            logger.debug(e)
            return None

    apptest.run = run

    yield apptest


@pytest.fixture
def safe_apptest_ss(safe_apptest: AppTest, session_state: SessionState) -> ty.Iterator[AppTest]:
    """Safe app for testing with bound session state."""
    session_state.bind(safe_apptest.session_state)
    yield safe_apptest


# Create a subclass of MagicMock that is recognized as an instance of OpenAI
class OpenAIMock(MagicMock):
    """A subclass of MagicMock that is recognized as an instance of `OpenAI`."""

    def __instancecheck__(self, instance: ty.Any) -> bool:
        return isinstance(instance, OpenAI)

    def __init__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        super().__init__(*args, **kwargs)
        # Mock method that returns LLM's response
        choice = MagicMock()
        choice.message.content = "Hi."
        response = MagicMock()
        response.choices = MagicMock(return_value=[MagicMock(return_value=choice)])
        # Now create the OpenAI mock
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(return_value=response)


@pytest.fixture
def mock_openai() -> ty.Iterator[MagicMock]:
    """Mock `OpenAI` client."""
    client = OpenAIMock(spec=OpenAI)
    assert isinstance(client, OpenAI)
    with patch.object(OpenAI, "__new__", return_value=client) as openaimock:
        yield openaimock


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
