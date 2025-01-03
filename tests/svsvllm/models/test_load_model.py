import pytest
from loguru import logger
import typing as ty
import sys, os

import torch
from transformers import BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase
from svsvllm.models import load_model


@pytest.mark.parametrize(
    "quantize, quantize_w_torch, load_in_4bit",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
    ],
)
def test_load_model(
    bnb_config: BitsAndBytesConfig,
    device: torch.device,
    quantize: bool,
    quantize_w_torch: bool,
    load_in_4bit: bool,
    tiny_llama_model_id: str,
) -> None:
    """Test `load_model`.

    Here, we test that this function is able to load a model and then we're able to use it.
    """
    # Load (quantized) model
    model, tokenizer = load_model(
        tiny_llama_model_id,
        bnb_config=bnb_config,
        device=device,
        quantize=quantize,
        quantize_w_torch=quantize_w_torch,
        load_in_4bit=load_in_4bit,
    )

    # Test
    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
