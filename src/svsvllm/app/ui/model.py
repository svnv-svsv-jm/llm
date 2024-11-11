__all__ = ["create_chat_model"]

import typing as ty
from loguru import logger
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import transformers
import torch

from svsvllm.loaders import load_model
from svsvllm.utils import find_device, add_chat_template
from svsvllm.app.ui.session_state import SessionState
from svsvllm.app.settings import settings


@st.cache_resource
def create_chat_model(
    model_name: str = None,
    apply_chat_template: bool = None,
    chat_template: str | dict[str, ty.Any] | list[dict[str, ty.Any]] | None = None,
    force_chat_template: bool = None,
    device: torch.device = None,
    pipeline_kwargs: dict[str, ty.Any] | None = None,
    **kwargs: ty.Any,
) -> ChatHuggingFace:
    """Create a chat model.

    Args:
        model_name (str, optional):
            Name of the model to use/download.
            For example: `"meta-llama/Meta-Llama-3.1-8B-Instruct"`.
            For all models, see HuggingFace website.
            Defaults to `None` (chosen from session state).

        apply_chat_template (bool, optional):
            Whether to apply chat template to tokenizer.
            Defaults to `None` (chosen from settings).

        chat_template (str | dict[str, ty.Any] | list[dict[str, ty.Any]]):
            Chat template to enforce when a default one is not available.
            Defaults to `None` (chosen from settings).

        force_chat_template (bool):
            If `True`, the provided chat template will be forced on the tokenizer.
            Defaults to `False`.

        device (torch.device):
            Device. Defaults to `None`.

        pipeline_kwargs (dict, optional):
            See :class:`HuggingFacePipeline`.

        **kwargs (Any):
            See :func:`load_model`.

    Returns:
        ChatHuggingFace: chat model.
    """
    logger.trace("Creating chat model")
    # Sanitize inputs
    if model_name is None:
        model_name = SessionState().state.model_name
    if apply_chat_template is None:
        apply_chat_template = settings.apply_chat_template
    if chat_template is None:
        chat_template = settings.chat_template
    if force_chat_template is None:
        force_chat_template = settings.force_chat_template
    if device is None:
        device = find_device()

    # Set input params
    kwargs.setdefault("quantize", SessionState().state.quantize)
    kwargs.setdefault("quantize_w_torch", SessionState().state.quantize_w_torch)

    # Load this model
    logger.trace(f"Loading model ({device}): {model_name}")
    hf_model, tokenizer = load_model(model_name, device=device, **kwargs)

    # Apply tokenizer template?
    tokenizer = add_chat_template(
        tokenizer,
        chat_template=chat_template,
        apply_chat_template=apply_chat_template,
        force_chat_template=force_chat_template,
    )

    # Save in state
    logger.trace(f"Loaded model: {model_name}")
    st.session_state["hf_model"] = hf_model
    st.session_state["tokenizer"] = tokenizer

    # Creating pipeline
    logger.trace(f"Creating `transformers.pipeline`")
    pipeline = transformers.pipeline(
        task="text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map=device,
        # device=device,
    )
    logger.trace(f"Created: {pipeline}")

    # Create HuggingFacePipeline
    logger.trace(f"Creating {HuggingFacePipeline}")
    logger.trace(f"pipeline_kwargs: {pipeline_kwargs}")
    llm = HuggingFacePipeline(pipeline=pipeline, model_id=model_name, pipeline_kwargs=pipeline_kwargs)
    logger.trace(f"Created: {llm}")

    # Create and save chat model
    logger.trace(f"Creating {ChatHuggingFace}")
    chat_model = ChatHuggingFace(llm=llm, model_id=model_name)
    st.session_state["chat_model"] = chat_model
    logger.trace(f"Chat model created: {chat_model}")

    # NOTE: This is necessary probably due to a bug in `HuggingFacePipeline`
    setattr(chat_model, "tokenizer", tokenizer)

    return chat_model
