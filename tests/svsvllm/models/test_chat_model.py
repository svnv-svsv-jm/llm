import pytest
from unittest.mock import patch
from loguru import logger
import typing as ty
import sys, os

import streamlit as st
from streamlit.testing.v1 import AppTest
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage
import torch
from transformers import BitsAndBytesConfig, Pipeline, PreTrainedTokenizerBase


from svsvllm.app.ui.model import create_chat_model
from svsvllm.defaults import DEFAULT_LLM, ZEPHYR_CHAT_TEMPLATE, CUSTOM_CHAT_TEMPLATE
from svsvllm.utils import CommandTimer


MESSAGES = [
    SystemMessage(content="You're a helpful assistant."),
    HumanMessage(content="What happens when an unstoppable force meets an immovable object?"),
]


@pytest.mark.parametrize("model_name", [DEFAULT_LLM])
@pytest.mark.parametrize("query", ["yo", MESSAGES])
@pytest.mark.parametrize("apply_chat_template", [True])
@pytest.mark.parametrize("chat_template", [ZEPHYR_CHAT_TEMPLATE])
def test_create_chat_model(
    apptest_ss: AppTest,  # Import just to bind session state
    bnb_config: BitsAndBytesConfig,
    device: torch.device,
    model_name: str,
    query: str,
    apply_chat_template: bool,
    chat_template: str | dict[str, ty.Any] | list[dict[str, ty.Any]],
) -> None:
    """Test we can query the created model."""
    # Create params for generation
    pipeline_kwargs: dict[str, ty.Any] = dict(
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        repetition_penalty=1.5,
        max_length=50,
    )
    # Create chat model
    chat_model = create_chat_model(
        model_name,
        apply_chat_template=apply_chat_template,
        chat_template=chat_template,
        pipeline_kwargs=pipeline_kwargs,
        device=device,
        bnb_config=bnb_config,
        quantize=True,
        quantize_w_torch=True,
    )
    logger.info(f"Chat model: {chat_model}")
    assert isinstance(chat_model, ChatHuggingFace)

    # Get LLM
    llm = chat_model.llm
    logger.info(f"LLM: {llm}")
    assert isinstance(llm, HuggingFacePipeline)

    # Get pipeline
    pipeline = llm.pipeline
    logger.info(f"Pipeline: {pipeline}")
    assert isinstance(pipeline, Pipeline)
    logger.info(f"Pipeline (model): {pipeline.model}")

    # Test tokenizers exist
    assert isinstance(pipeline.tokenizer, PreTrainedTokenizerBase)
    # Test tokenizer: Pipeline
    if apply_chat_template:
        assert pipeline.tokenizer.chat_template is not None

    # Invoke pipeline
    if not isinstance(query, str):
        pytest.skip(f"Cannot run `pipeline` with type {type(query)}")
    with CommandTimer("pipeline"):
        msg = pipeline(query, **pipeline_kwargs)

    # Invoke
    with CommandTimer("chat_model"):
        msg = chat_model.invoke(query, pipeline_kwargs=pipeline_kwargs)

    logger.success(f"Response: {msg}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
