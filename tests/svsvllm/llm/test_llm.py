import pytest
from unittest.mock import patch
from loguru import logger
import typing as ty
import sys, os
import yaml

import torch
from transformers import BitsAndBytesConfig
from langchain_community.vectorstores.faiss import FAISS

from svsvllm.utils import CommandTimer
from svsvllm.loaders import llm_chain, load_model
from svsvllm.rag import ITALIAN_PROMPT_TEMPLATE


@pytest.mark.parametrize(
    "model_name, quantize, quantize_w_torch, max_new_tokens",
    [
        ("BEE-spoke-data/smol_llama-101M-GQA", True, True, 100),
    ],
)
def test_llm(
    artifact_location: str,
    database: FAISS,
    bnb_config: BitsAndBytesConfig,
    patch_torch_quantized_engine: bool,
    device: torch.device,
    model_name: str,
    quantize: bool,
    quantize_w_torch: bool,
    max_new_tokens: int,
) -> None:
    """Test we can run a simple example.

    Args:
        artifact_location (str):
            See `conftest.py`.

        database (FAISS):
            See `conftest.py`.

        bnb_config (BitsAndBytesConfig):
            See `conftest.py`.

        patch_torch_quantized_engine (bool):
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
    # Load model
    model, tokenizer = load_model(
        model_name,
        bnb_config=bnb_config,
        quantize=quantize,
        quantize_w_torch=quantize_w_torch,
    )

    # Question
    question = "come si calcola la plusvalenza sulla cessione di criptoattività?"
    logger.info(f"Invoking LLM with question: {question}")

    # LLM's
    logger.info("Creating LLMs...")
    chain = llm_chain(
        model,
        tokenizer,
        prompt_template=ITALIAN_PROMPT_TEMPLATE,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    chain_w_rag = llm_chain(
        model,
        tokenizer,
        database=database,
        prompt_template=ITALIAN_PROMPT_TEMPLATE,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    # Run LLM's
    logger.info("Running LLMs...")
    with CommandTimer(f"{model_name} (no-rag)"):
        answer_no_rag = chain.invoke({"context": "", "question": question})
    with CommandTimer(f"{model_name} (with-rag)"):
        answer_w_rag = chain_w_rag.invoke(question)

    # Save answers
    logger.info("Saving LLMs' answers...")
    name = model_name.replace("/", "--")
    answers = {name: dict(no_rag=answer_no_rag, rag=answer_w_rag)}
    with open(os.path.join(artifact_location, f"{name}.yaml"), "w") as outfile:
        yaml.dump(answers, outfile, indent=2)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
