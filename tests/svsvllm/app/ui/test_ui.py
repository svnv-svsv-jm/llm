import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest

# TODO: see https://medium.com/@chrisschneider/build-a-high-quality-streamlit-app-with-test-driven-development-eef4e462f65e


def test_app_starts(apptest: AppTest) -> None:
    """Verify the app starts without errors."""
    apptest.run()
    assert not apptest.exception


def test_page_title(apptest: AppTest) -> None:
    """Verify the app has the expected title."""
    apptest.session_state["has_chat"] = False
    apptest.run()
    assert len(apptest.subheader) == 1
    assert "Smart Assistant" in apptest.subheader[0].value
    assert len(apptest.caption) > 0
    assert len(apptest.title) > 0
    assert "FiscalAI" in apptest.title[0].value


def test_shows_welcome_message(apptest: AppTest) -> None:
    """Verify initial message from assistant is shown."""
    apptest.session_state["has_chat"] = False
    apptest.run()
    assert len(apptest.chat_message) == 1
    assert apptest.chat_message[0].name == "assistant"
    assert "help" in apptest.chat_message[0].markdown[0].value


@pytest.mark.parametrize("openai_api_key", [None, "any"])
@pytest.mark.parametrize("state_for_client", [False, True])
def test_openai(
    apptest: AppTest,
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
    """Test app is ready to work with OpenAI model.

    Args:
        state_for_client (bool):
            If `True`, we inject the mock client into the state. If `False`, we create a new `OpenAI` client, but still the `OpenAI.__new__` method is mocked.

        openai_api_key (str | None):
            When `None`, the open source model will be used.
    """
    # Inject mock into app state to avoid creation of OpenAI client
    if state_for_client:
        apptest.session_state["openai_client"] = mock_openai

    # Inject key
    apptest.session_state["openai_api_key"] = openai_api_key
    # Inject fake model name
    apptest.session_state["model_name"] = "TinyLlama/TinyLlama_v1.1"

    # Run app with mocked user inputs
    apptest.run()
    mock_chat_input.assert_called()
    if openai_api_key is None:
        # mock_transformers_pipeline.assert_called()
        # mock_hf_pipeline.assert_called()
        # mock_hf_chat.assert_called()
        mock_agent_stream.assert_called()

    # Test: OpenAI key
    assert apptest.session_state.openai_api_key == openai_api_key

    # Test: chat history exists
    logger.info(apptest.chat_message)
    assert len(apptest.chat_message) == 3
    assert len(apptest.session_state.messages) == 3


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
