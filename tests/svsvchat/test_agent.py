import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

from codetiming import Timer
from streamlit.testing.v1 import AppTest
from langchain_core.vectorstores import VectorStoreRetriever

from svsvllm.exceptions import NoChatModelError
from svsvchat.settings import Settings
from svsvchat.session_state import SessionState
from svsvchat.rag import create_history_aware_retriever
from svsvchat.agent import create_agent


@pytest.mark.parametrize("create_chat", [False, True])
def test_create_agent(
    settings: Settings,
    session_state: SessionState,
    model_id: str,
    retriever: VectorStoreRetriever,
    create_chat: bool,
) -> None:
    """Test `create_agent`."""
    with patch.object(session_state, "retriever", retriever):
        if create_chat:
            create_history_aware_retriever(model_name=model_id)

        # If no chat model, this is not possible
        if session_state.chat_model is None:
            with pytest.raises(NoChatModelError):
                create_agent()
            return

        # If chat model is present, we can create the agent
        agent = create_agent()
    logger.success(f"Agent: {agent}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
