import pytest
from unittest.mock import patch, Mock, MagicMock
from loguru import logger
import typing as ty
import sys, os

from codetiming import Timer
from streamlit.testing.v1 import AppTest
from langchain_core.messages import AIMessage
from langchain_huggingface import ChatHuggingFace

from svsvchat.const import DEFAULT_LLM_MLX
from svsvchat.session_state import SessionState
from svsvchat.settings import settings


@pytest.mark.parametrize("openai_api_key", [None, "any"])
def test_ui(
    session_state: SessionState,
    apptest: AppTest,
    res_docs_path: str,
    mock_openai: MagicMock,
    mock_chat_input: MagicMock,
    mock_agent_stream: MagicMock,
    openai_api_key: str | None,
) -> None:
    """Test app is ready to work with or without OpenAI model.

    Args:
        openai_api_key (str | None):
            When `None`, the open source model will be used.
    """
    # Inject key
    session_state.openai_api_key = openai_api_key
    apptest.session_state.openai_api_key = openai_api_key

    # Run app with mocked user inputs
    with patch.object(settings, "uploaded_files_dir", res_docs_path):
        # NOTE: we may even `apptest.chat_input[0].set_value("Hi").run()` but we have to run first once
        with Timer("apptest.run"):
            apptest.run()

    # Test: OpenAI key
    if openai_api_key is not None:
        assert session_state.openai_api_key is not None
        assert apptest.session_state.openai_api_key is not None

    # Test HF model name exits regardless
    assert session_state.model_name == DEFAULT_LLM_MLX

    # Test no erros were raised
    for i, ex in enumerate(apptest.exception):
        logger.info(f"({i}): {ex}")
    assert len(apptest.exception) == 0

    # End
    logger.success("PASSED")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
