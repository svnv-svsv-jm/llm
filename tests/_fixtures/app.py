# pylint: disable=reimported
__all__ = [
    "app_main_file",
    "apptest",
    "apptest_ss",
    "safe_apptest",
    "safe_apptest_ss",
    "session_state",
    "mock_openai",
    "mock_chat_input",
    "mock_agent_stream",
    "mock_text_file",
    "mock_rag_docs",
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
from svsvllm.app.settings import settings
from svsvllm.app.ui.session_state import SessionState


@pytest.fixture
def mock_rag_docs(res_docs_path: str) -> ty.Iterator[str]:
    """Run app with mocked user inputs."""
    with patch.object(settings, "uploaded_files_dir", res_docs_path) as mock:
        yield mock


@pytest.fixture
def session_state() -> ty.Iterator[SessionState]:
    """Session state."""
    with SessionState(reverse=True, auto_sync=False) as ss:
        st.cache_resource.clear()
        yield ss


@pytest.fixture
def app_main_file() -> str:
    """App file."""
    path = os.path.abspath(main.__file__)
    logger.debug(f"Loading script: {path}")
    return path


@pytest.fixture
def apptest(trace_logging_level: bool, app_main_file: str) -> ty.Iterator[AppTest]:
    """App for testing."""
    with patch.object(settings, "test_mode", True):
        at = AppTest.from_file(app_main_file, default_timeout=30)
        yield at


# NOTE: Binding to `apptest.session_state` causes `apptest.run()` to hang until timeout. This may be due to `AppTest` not being able to exit threads due to `SessionState` still referencing it
@pytest.fixture
def apptest_ss(apptest: AppTest, session_state: SessionState) -> ty.Iterator[AppTest]:
    """App for testing with bound session state."""
    session_state.bind(st.session_state)
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

    apptest.run = run  # type: ignore

    yield apptest


@pytest.fixture
def safe_apptest_ss(safe_apptest: AppTest, session_state: SessionState) -> ty.Iterator[AppTest]:
    """Safe app for testing with bound session state."""
    session_state.bind(st.session_state)
    yield safe_apptest


# Create a subclass of MagicMock that is recognized as an instance of OpenAI
class OpenAIMock(MagicMock):
    """A subclass of MagicMock that is recognized as an instance of `OpenAI`."""

    def __init__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        """Init superclass with `spec=OpenAI`, then create necessary attributes."""
        super().__init__(*args, spec=OpenAI, **kwargs)
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
    client = OpenAIMock()
    assert isinstance(client, OpenAI)
    with patch.object(OpenAI, "__new__", return_value=client) as openaimock:
        yield openaimock


@pytest.fixture
def mock_chat_input() -> ty.Iterator[MagicMock]:
    """Mock first few user inputs."""
    with patch.object(st, "chat_input", side_effect=["Hello", None]) as chat_input:
        yield chat_input


@pytest.fixture
def mock_agent_stream(request: pytest.FixtureRequest) -> ty.Iterator[MagicMock]:
    """Mock agent's `CompiledGraph.stream()` method.
    We create a mock `dict` object with the `"messages"` key having a `list[AIMessage]` value.
    """
    if hasattr(request, "param"):
        msg = f"{request.param}"
    else:
        msg = "This is a mocked message."
    # Create an object to stream
    stream = MagicMock(return_value={"messages": [AIMessage(content=msg)]})
    # Patch and pass a list `side_effect` to simulate the `yield` effect (streaming)
    with patch.object(CompiledGraph, "stream", side_effect=[stream] * 5) as mock_stream:
        yield mock_stream


@pytest.fixture
def mock_text_file() -> BytesIO:
    """Mock file content (can be any file type you're expecting)."""
    mock_file = BytesIO(b"Test file content")
    mock_file.name = "test-file.txt"
    return mock_file
