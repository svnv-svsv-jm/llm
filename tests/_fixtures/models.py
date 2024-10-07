import pytest
import typing as ty
from loguru import logger

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer,
    MixtralForCausalLM,
)
from optimum.quanto import QuantizedModelForCausalLM

from svsvllm.loaders import load_model


@pytest.fixture
def tiny_llama(
    bnb_config: BitsAndBytesConfig | None,
) -> tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """TinyLlama."""
    model_name = "TinyLlama/TinyLlama_v1.1"
    model, tokenizer = load_model(
        model_name,
        bnb_config=bnb_config,
        quantize=True,
        quantize_w_torch=True,
    )
    return model, tokenizer


@pytest.fixture
def cerbero(
    bnb_config: BitsAndBytesConfig | None,
) -> ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """Cerbero."""
    model_name = "galatolo/cerbero-7b"  # Italian
    model, tokenizer = load_model(
        model_name,
        bnb_config=bnb_config,
        quantize=True,
        quantize_w_torch=False,
    )
    return model, tokenizer


@pytest.fixture
def mistral_small(
    bnb_config: BitsAndBytesConfig | None,
) -> ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """Small mistral."""
    model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    model, tokenizer = load_model(
        model_name,
        bnb_config=bnb_config,
        quantize=True,
        quantize_w_torch=False,
        model_class=MixtralForCausalLM,
        tokenizer_class=LlamaTokenizer,
    )
    return model, tokenizer


@pytest.fixture
def mistral7b(
    bnb_config: BitsAndBytesConfig | None,
) -> ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """Small mistral."""
    model_name = "mistralai/Mistral-7B-v0.1"
    model, tokenizer = load_model(
        model_name,
        bnb_config=bnb_config,
        quantize=True,
        quantize_w_torch=True,
    )
    return model, tokenizer
