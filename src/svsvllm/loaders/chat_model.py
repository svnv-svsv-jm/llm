__all__ = ["load_chat_model"]

import typing as ty
from loguru import logger

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import transformers
import torch

from svsvllm.loaders import load_model
from svsvllm.utils import find_device, add_chat_template
from svsvllm.defaults import ZEPHYR_CHAT_TEMPLATE as CHAT_TEMPLATE


def load_chat_model(
    model_name: str,
    apply_chat_template: bool = None,
    chat_template: str | dict[str, ty.Any] | list[dict[str, ty.Any]] = CHAT_TEMPLATE,
    force_chat_template: bool = False,
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
    if device is None:
        device = find_device()
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

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

    # Debug
    logger.trace(f"Loaded model: {model_name}")

    # Creating pipeline
    logger.trace(f"Creating `transformers.pipeline`")
    pipeline = transformers.pipeline(
        task="text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map=device,
        **pipeline_kwargs,
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
    logger.trace(f"Chat model created: {chat_model}")

    # NOTE: This is necessary probably due to a bug in `HuggingFacePipeline`
    setattr(chat_model, "tokenizer", tokenizer)

    return chat_model
