import pytest
from unittest.mock import patch, Mock, MagicMock
from loguru import logger
import typing as ty
import sys, os

import streamlit as st
from streamlit.testing.v1 import AppTest
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import torch

from svsvllm.app.ui.agent import create_agent
from svsvllm.app.ui.rag import create_history_aware_retriever
from svsvllm.app.settings import settings
from svsvllm.app.ui.session_state import SessionState


def test_create_agent(
    apptest_ss: AppTest,
    mock_rag_docs: str,
    device: torch.device,
    pipeline_kwargs: dict,
) -> None:
    """Create history aware retriever and agent, then test creations are as expected."""
    # Create history aware retriever
    create_history_aware_retriever(pipeline_kwargs=pipeline_kwargs)

    # Create agent
    agent = create_agent()
    logger.info(f"Agent: {agent}")

    # Get chat model
    state = SessionState().state
    chat_model = state.chat_model
    assert isinstance(chat_model, ChatHuggingFace)

    # Get LLM
    llm = chat_model.llm
    logger.info(f"LLM: {llm}")
    assert isinstance(llm, HuggingFacePipeline)
    assert llm.pipeline_kwargs == pipeline_kwargs, f"Expected {pipeline_kwargs} but got {llm.pipeline_kwargs}"


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
