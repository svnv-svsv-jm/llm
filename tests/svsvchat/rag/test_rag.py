import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

from svsvchat.rag import initialize_rag, initialize_retriever
from svsvchat.session_state import SessionState
from svsvchat.settings import Settings


def test_initialize_retriever(
    session_state: SessionState,
    settings: Settings,
    res_docs_path: str,
) -> None:
    """Test `initialize_retriever`."""
    assert session_state.retriever is None
    assert session_state.db is None
    with patch.object(settings, "uploaded_files_dir", res_docs_path):
        initialize_retriever()
        # Call twice to check early return is hit because retriever exists already
        initialize_retriever()
    # Tests
    assert session_state.retriever is not None
    assert session_state.db is not None


def test_initialize_rag(
    session_state: SessionState,
    settings: Settings,
    res_docs_path: str,
) -> None:
    """Create a RAG with the history aware retriever, then test their creation happened as expected."""
    # Create history aware retriever
    assert session_state.retriever is None
    assert session_state.db is None
    with patch.object(settings, "uploaded_files_dir", res_docs_path):
        initialize_rag()

        # Call twice to check early return is hit because retriever exists already
        initialize_rag()

    # Test
    assert session_state.retriever is not None
    assert session_state.db is not None
    assert session_state.history_aware_retriever is not None


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
