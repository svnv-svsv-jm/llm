import pytest
from loguru import logger
import typing as ty
import sys, os
import yaml

import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import transformers

from svsvllm.loaders import load_model
from svsvllm.utils import CommandTimer


@pytest.mark.parametrize(
    "model_name, quantize, quantize_w_torch, model_class, tokenizer_class",
    [
        ("TinyLlama/TinyLlama_v1.1", True, False, None, None),
        ("TinyLlama/TinyLlama_v1.1", True, True, None, None),
        ("BEE-spoke-data/smol_llama-101M-GQA", True, False, None, None),
        ("BEE-spoke-data/smol_llama-101M-GQA", True, True, None, None),
    ],
)
def test_model_loader(
    artifact_location: str,
    bnb_config: BitsAndBytesConfig,
    device: torch.device,
    model_name: str,
    quantize: bool,
    quantize_w_torch: bool,
    model_class: type[AutoModelForCausalLM] | None,
    tokenizer_class: type[AutoTokenizer] | None,
) -> None:
    """Test model loader.
    Args:
        artifact_location (str):
            See `conftest.py`.

        bnb_config (BitsAndBytesConfig):
            See `conftest.py`.

        device (torch.device):
            See `conftest.py`.

        model_name (str):
            See `svsvllm.loaders.load_model`.

        quantize (bool):
            See `svsvllm.loaders.load_model`.

        quantize_w_torch (bool):
            See `svsvllm.loaders.load_model`.

        model_class (type[AutoModelForCausalLM] | None):
            See `svsvllm.loaders.load_model`.

        tokenizer_class (type[AutoTokenizer] | None):
            See `svsvllm.loaders.load_model`.
    """
    # Load (quantized) model
    model, tokenizer = load_model(
        model_name,
        bnb_config=bnb_config,
        quantize=quantize,
        quantize_w_torch=quantize_w_torch,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
    )

    # Create pipeline
    pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map=device,
    )

    # Run
    with CommandTimer(model_name):
        sequences = pipeline(
            "Tell me who you are.",
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            repetition_penalty=1.5,
            eos_token_id=tokenizer.eos_token_id,
            max_length=500,
        )

    # Log generated text
    answer = ""
    for seq in sequences:
        s = seq["generated_text"]
        logger.info(f"Result: {s}")
        answer += f"{s}"

    # Save answers
    logger.info("Saving LLMs' answers...")
    name = model_name.replace("/", "--")
    answers = {name: answer}
    with open(os.path.join(artifact_location, f"{name}.yaml"), "w") as outfile:
        yaml.dump(answers, outfile, indent=2)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
