import pytest
from loguru import logger
import typing as ty
import sys, os
import yaml

import torch
from transformers import BitsAndBytesConfig
from langchain_core.messages import HumanMessage

from svsvllm.loaders import load_chat_model


@pytest.mark.parametrize(
    "quantize, quantize_w_torch, load_in_4bit, use_mlx",
    [
        (False, False, False, True),
        (True, False, False, False),
        (True, True, False, False),
    ],
)
def test_load_model(
    artifact_location: str,
    bnb_config: BitsAndBytesConfig,
    device: torch.device,
    quantize: bool,
    quantize_w_torch: bool,
    use_mlx: bool,
    load_in_4bit: bool,
    pipeline_kwargs: dict[str, ty.Any],
    mlx_model_id: str,
    model_id: str,
) -> None:
    """Test `load_chat_model`.

    Here, we test that this function is able to load a model and then we're able to use it.
    """
    model_name = mlx_model_id if use_mlx else model_id
    # Load (quantized) model
    chat, _, _ = load_chat_model(
        model_name,
        bnb_config=bnb_config,
        device=device,
        quantize=quantize,
        quantize_w_torch=quantize_w_torch,
        pipeline_kwargs=pipeline_kwargs,
        use_mlx=use_mlx,
        load_in_4bit=load_in_4bit,
    )

    # Messsages
    messages = [HumanMessage(content="What happens when an unstoppable force meets an immovable object?")]

    if not use_mlx:
        return

    # Invoke
    answer = chat.invoke(messages)

    # Save answers
    logger.info("Saving LLMs' answers...")
    name = model_name.replace("/", "--")
    answers = {name: answer.content}
    with open(os.path.join(artifact_location, f"{name}.yaml"), "w") as outfile:
        yaml.dump(answers, outfile, indent=2)

    logger.success(f"({type(answer)}): {answer}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
