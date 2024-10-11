import pytest
import typing as ty
from loguru import logger

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline
from optimum.quanto import QuantizedModelForCausalLM


@pytest.fixture
def tiny_llama_pipeline(
    tiny_llama: tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer],
    device: torch.device,
) -> Pipeline:
    """TinyLlama LLM."""
    model, tokenizer = tiny_llama
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map=device,
    )
