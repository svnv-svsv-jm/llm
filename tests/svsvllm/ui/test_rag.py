import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

from streamlit.testing.v1 import AppTest
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage

from svsvllm.ui.rag import initialize_rag, create_history_aware_retriever
from svsvllm.ui.session_state import SessionState
from svsvllm.types import ChatModelType


def test_ui_rag(
    apptest_ss: AppTest,
    mock_rag_docs: str,
    device: torch.device,
    pipeline_kwargs: dict,
    query: str,
    mlx_model_id: str,
) -> None:
    """Create a RAG with the history aware retriever, then test their creation happened as expected."""
    # Create history aware retriever
    retriever = initialize_rag()
    logger.info(f"retriever: {retriever}")
    history_aware_retriever = create_history_aware_retriever(
        pipeline_kwargs=pipeline_kwargs,
        model_name=mlx_model_id,
        use_mlx=True,
    )
    logger.info(f"history_aware_retriever: {history_aware_retriever}")

    # Get chat model
    state = SessionState().state
    chat_model = state.chat_model
    assert isinstance(chat_model, ChatModelType)

    # Get LLM
    llm = getattr(chat_model, "llm", None)
    logger.info(f"LLM: {llm}")
    if isinstance(llm, HuggingFacePipeline):
        assert llm.pipeline_kwargs, f"Expected {pipeline_kwargs} but got {llm.pipeline_kwargs}"
        assert (
            llm.pipeline_kwargs == pipeline_kwargs
        ), f"Expected {pipeline_kwargs} but got {llm.pipeline_kwargs}"
    else:
        messages = [HumanMessage(content=query)]
        answer = chat_model.invoke(messages)
        logger.success(f"{answer.content}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
