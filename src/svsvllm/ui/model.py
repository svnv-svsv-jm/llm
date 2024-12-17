__all__ = ["create_chat_model"]

import typing as ty
from loguru import logger

import streamlit as st

from svsvllm.loaders import load_chat_model
from svsvllm.ui.session_state import session_state
from svsvllm.settings import settings
from svsvllm.types import ChatModelType


@st.cache_resource
def create_chat_model(model_name: str = None, **kwargs: ty.Any) -> ChatModelType:
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
        ChatModelType: chat model.
    """
    # Sanitize inputs
    if model_name is None:
        model_name = session_state.model_name

    logger.trace(f"Creating chat model: {model_name}")

    # Set input params
    kwargs.setdefault("apply_chat_template", settings.apply_chat_template)
    kwargs.setdefault("chat_template", settings.chat_template)
    kwargs.setdefault("force_chat_template", settings.force_chat_template)
    kwargs.setdefault("quantize", session_state.quantize)
    kwargs.setdefault("quantize_w_torch", session_state.quantize_w_torch)
    kwargs.setdefault("use_mlx", session_state.use_mlx)

    chat_model, model, tokenizer = load_chat_model(model_name, **kwargs)

    # Save in state
    logger.trace(f"Loaded model: {model_name}")
    session_state.model = model
    session_state.tokenizer = tokenizer
    session_state.chat_model = chat_model
    logger.trace(f"Chat model created: {chat_model}")

    return chat_model
