import pytest
from loguru import logger
import typing as ty
import sys, os
import yaml

import torch
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    MixtralForCausalLM,
)
import transformers

from svsvllm.loaders import load_model
from svsvllm.utils import CommandTimer


@pytest.mark.parametrize(
    "model_name, quantize, quantize_w_torch, model_class, tokenizer_class",
    [
        # (
        #     "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",  # Too big...
        #     True,
        #     True,
        #     MixtralForCausalLM,
        #     LlamaTokenizer,
        # ),
        ("TinyLlama/TinyLlama_v1.1", True, False, None, None),
        ("BEE-spoke-data/smol_llama-101M-GQA", True, False, None, None),
    ],
)
def test_model_loader(
    model_name: str,
    bnb_config: BitsAndBytesConfig,
    quantize: bool,
    quantize_w_torch: bool,
    device: torch.device,
    model_class: type[AutoModelForCausalLM] | None,
    tokenizer_class: type[AutoTokenizer] | None,
    artifact_location: str,
) -> None:
    """Test model loader."""
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
