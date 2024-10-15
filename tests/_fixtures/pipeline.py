__all__ = ["default_llm_pipeline"]

import pytest
import typing as ty
from loguru import logger

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline
from optimum.quanto import QuantizedModelForCausalLM


@pytest.fixture
def default_llm_pipeline(
    default_llm: tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer],
    device: torch.device,
) -> Pipeline:
    """TinyLlama LLM."""
    model, tokenizer = default_llm
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map=device,
    )
