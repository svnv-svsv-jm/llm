__all__ = ["create_chat_model"]

import typing as ty
from loguru import logger

import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import Pipeline

from svsvllm.loaders import load_chat_model
from svsvllm.ui.session_state import SessionState
from svsvllm.settings import settings


@st.cache_resource
def create_chat_model(model_name: str = None, **kwargs: ty.Any) -> ChatHuggingFace:
    """Create a chat model.

    Args:
        model_name (str, optional):
            Name of the model to use/download.
            For example: `"meta-llama/Meta-Llama-3.1-8B-Instruct"`.
            For all models, see HuggingFace website.
            Defaults to `None` (chosen from session state).

        **kwargs (Any):
            See :func:`load_chat_model`.

    Returns:
        ChatHuggingFace: chat model.
    """
    state = SessionState().state
    # Sanitize inputs
    if model_name is None:
        model_name = state.model_name

    logger.trace(f"Creating chat model: {model_name}")

    # Set input params
    kwargs.setdefault("apply_chat_template", settings.apply_chat_template)
    kwargs.setdefault("chat_template", settings.chat_template)
    kwargs.setdefault("force_chat_template", settings.force_chat_template)
    kwargs.setdefault("quantize", SessionState().state.quantize)
    kwargs.setdefault("quantize_w_torch", SessionState().state.quantize_w_torch)

    chat_model = load_chat_model(model_name, **kwargs)

    # Get LLM
    llm: HuggingFacePipeline = chat_model.llm
    logger.trace(f"LLM: {llm}")

    # Get pipeline
    pipeline: Pipeline = llm.pipeline
    logger.trace(f"Pipeline: {pipeline}")
    logger.trace(f"Pipeline (model): {pipeline.model}")

    # Save in state
    logger.trace(f"Loaded model: {model_name}")
    state.hf_model = pipeline.model
    state.tokenizer = pipeline.tokenizer

    state.chat_model = chat_model
    logger.trace(f"Chat model created: {chat_model}")

    return chat_model
