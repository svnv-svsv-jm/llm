import pytest
from loguru import logger
import typing as ty
import sys, os
import yaml

import torch
from transformers import BitsAndBytesConfig
import transformers

from svsvllm.loaders import load_model
from svsvllm.utils import CommandTimer
from svsvllm.defaults import DEFAULT_LLM


@pytest.mark.parametrize(
    "model_name, quantize, quantize_w_torch",
    [
        (DEFAULT_LLM, True, False),
        (DEFAULT_LLM, True, True),
    ],
)
def test_load_model(
    artifact_location: str,
    bnb_config: BitsAndBytesConfig,
    device: torch.device,
    model_name: str,
    quantize: bool,
    quantize_w_torch: bool,
) -> None:
    """Test `load_model`. This function loads models from HuggingFace, but also helps quantizing them.
    Here, we test that this function is able to quantize a model and then we're able to use it.

    Args:
        model_name (str):
            See `svsvllm.loaders.load_model`.

        quantize (bool):
            See `svsvllm.loaders.load_model`.

        quantize_w_torch (bool):
            See `svsvllm.loaders.load_model`.
    """
    # Load (quantized) model
    model, tokenizer = load_model(
        model_name,
        bnb_config=bnb_config,
        quantize=quantize,
        quantize_w_torch=quantize_w_torch,
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
