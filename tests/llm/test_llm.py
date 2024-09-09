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

# torch.backends.quantized.engine = 'qnnpack'


@pytest.mark.parametrize(
    "model_name, quantize, quantize_w_torch",
    [
        # ("BEE-spoke-data/smol_llama-101M-GQA", True, True),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", True, True),
        ("TinyLlama/TinyLlama_v1.1", True, True),
    ],
)
def test_llm(
    model_name: str,
    artifact_location: str,
    database: FAISS,
    bnb_config: BitsAndBytesConfig,
    quantize: bool,
    quantize_w_torch: bool,
    patch_torch_quantized_engine: bool,
    device: torch.device,
) -> None:
    """Test we can run a simple example."""
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
    llm = llm_chain(
        model,
        tokenizer,
        prompt_template=ITALIAN_PROMPT_TEMPLATE,
        device=device,
    )
    llm_w_rag = llm_chain(
        model,
        tokenizer,
        database=database,
        prompt_template=ITALIAN_PROMPT_TEMPLATE,
        device=device,
    )

    # Run LLM's
    logger.info("Running LLMs...")
    with CommandTimer(f"{model_name} (no-rag)"):
        answer_no_rag = llm.invoke({"context": "", "question": question})
    with CommandTimer(f"{model_name} (with-rag)"):
        answer_w_rag = llm_w_rag.invoke(question)

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
