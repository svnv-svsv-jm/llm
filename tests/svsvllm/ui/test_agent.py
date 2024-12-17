import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

import streamlit as st
from streamlit.testing.v1 import AppTest
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage
import torch

from svsvllm.ui.agent import create_agent
from svsvllm.ui.rag import create_history_aware_retriever
from svsvllm.ui.session_state import SessionState
from svsvllm.types import ChatModelType


@pytest.mark.parametrize("use_mlx", [True, False])
def test_create_agent(
    apptest_ss: AppTest,
    mock_rag_docs: str,
    device: torch.device,
    pipeline_kwargs: dict,
    use_mlx: bool,
    model_id: str,
    mlx_model_id: str,
) -> None:
    """Create history aware retriever and agent, then test creations are as expected."""
    model_name = mlx_model_id if use_mlx else model_id

    # Create history aware retriever
    create_history_aware_retriever(
        model_name=model_name,
        pipeline_kwargs=pipeline_kwargs,
        use_mlx=use_mlx,
    )

    # Create agent
    agent = create_agent()
    logger.info(f"Agent: {agent}")

    # Get chat model
    state = SessionState().state
    chat_model = state.chat_model
    assert isinstance(chat_model, ChatModelType)

    # Get LLM
    llm = getattr(chat_model, "llm", None)
    logger.info(f"LLM: {llm}")
    if isinstance(llm, HuggingFacePipeline):
        assert (
            llm.pipeline_kwargs == pipeline_kwargs
        ), f"Expected {pipeline_kwargs} but got {llm.pipeline_kwargs}"

    # Early stop?
    if not use_mlx:
        logger.success("No mlx.")
        return

    # Invoke
    messages = [HumanMessage(content="What happens when an unstoppable force meets an immovable object?")]
    answer = chat_model.invoke(messages)
    logger.success(f"use_mlx={use_mlx}: {answer.content}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
