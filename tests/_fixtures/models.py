__all__ = ["default_llm", "cerbero", "mistral_small", "mistral7b"]

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
from svsvllm.defaults import DEFAULT_LLM


@pytest.fixture
def llm(
    request: pytest.FixtureRequest,
    bnb_config: BitsAndBytesConfig | None,
) -> tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """Custom LLM.
    This can be only used as parametrized fixture, or it will throw an `AttributeError`.

    When this fixture is parametrized INDERECTLY, the parameters are actually received here as inputs (via `request.param`).

    Example usage:
    ```python
    @pytest.mark.parametrize('llm', ['mistralai/Mistral-7B-v0.1'], indirect=True)
    def test_dummy(llm: tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]) -> None:
        '''Every time this test runs, the current input `str` value is passed to the fixture, which loads the model and returns it.'''
        print(llm)
    ```
    """
    model_name = request.param
    model, tokenizer = load_model(
        model_name,
        bnb_config=bnb_config,
        quantize=True,
        quantize_w_torch=True,
    )
    return model, tokenizer


@pytest.fixture(scope="session")
def default_llm(
    bnb_config: BitsAndBytesConfig | None,
) -> tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """Default LLM."""
    model_name = DEFAULT_LLM
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
