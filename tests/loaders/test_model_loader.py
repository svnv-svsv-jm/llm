import pytest
from loguru import logger
import typing as ty
import sys, os

import torch
from transformers import BitsAndBytesConfig
import transformers

from svsvllm.loaders import load_model
from svsvllm.utils import CommandTimer


@pytest.mark.parametrize(
    "model_name, quantize, quantize_w_torch",
    [
        ("mistralai/Mistral-7B-v0.1", True, False),
        ("TinyLlama/TinyLlama_v1.1", True, False),
        ("BEE-spoke-data/smol_llama-101M-GQA", True, False),
    ],
)
def test_model_loader(
    model_name: str,
    bnb_config: BitsAndBytesConfig,
    quantize: bool,
    quantize_w_torch: bool,
    device: torch.device,
) -> None:
    """Test model loader."""
    model, tokenizer = load_model(
        model_name,
        bnb_config=bnb_config,
        quantize=quantize,
        quantize_w_torch=quantize_w_torch,
    )
    pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map=device,
    )
    with CommandTimer():
        sequences = pipeline(
            "This is a test.",
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            repetition_penalty=1.5,
            eos_token_id=tokenizer.eos_token_id,
            max_length=500,
        )
    for seq in sequences:
        logger.info(f"Result: {seq['generated_text']}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
