import pytest
from unittest.mock import patch, Mock, MagicMock
from loguru import logger
import typing as ty
import sys, os

import streamlit as st
from streamlit.testing.v1 import AppTest
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from svsvllm.ui.rag import initialize_rag, create_history_aware_retriever
from svsvllm.ui.session_state import SessionState


def test_ui_rag(
    apptest_ss: AppTest,
    mock_rag_docs: str,
    device: torch.device,
    pipeline_kwargs: dict,
) -> None:
    """Create a RAG with the history aware retriever, then test their creation happened as expected."""
    # Create history aware retriever
    retriever = initialize_rag()
    logger.info(f"retriever: {retriever}")
    history_aware_retriever = create_history_aware_retriever(pipeline_kwargs=pipeline_kwargs)
    logger.info(f"history_aware_retriever: {history_aware_retriever}")

    # Get chat model
    state = SessionState().state
    chat_model = state.chat_model
    assert isinstance(chat_model, ChatHuggingFace)

    # Get LLM
    llm = chat_model.llm
    logger.info(f"LLM: {llm}")
    assert isinstance(llm, HuggingFacePipeline)
    assert llm.pipeline_kwargs, f"Expected {pipeline_kwargs} but got {llm.pipeline_kwargs}"
    assert llm.pipeline_kwargs == pipeline_kwargs, f"Expected {pipeline_kwargs} but got {llm.pipeline_kwargs}"


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
