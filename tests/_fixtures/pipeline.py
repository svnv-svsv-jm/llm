__all__ = ["pipeline_kwargs", "default_llm_pipeline"]

import pytest
import typing as ty
from loguru import logger

import torch
from transformers import PreTrainedModel, pipeline, Pipeline, PreTrainedTokenizerBase


@pytest.fixture
def pipeline_kwargs() -> dict:
    """Pipeline kwargs for testing."""
    # Create params for generation
    pipeline_kwargs: dict[str, ty.Any] = dict(
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        repetition_penalty=1.5,
        max_length=50,
    )
    return pipeline_kwargs


@pytest.fixture
def default_llm_pipeline(
    default_llm: tuple[PreTrainedModel, PreTrainedTokenizerBase],
    device: torch.device,
    pipeline_kwargs: dict,
) -> Pipeline:
    """TinyLlama LLM."""
    model, tokenizer = default_llm
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map=device,
        **pipeline_kwargs
    )
