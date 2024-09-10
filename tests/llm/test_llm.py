import pytest
from unittest.mock import patch
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
from langchain_community.vectorstores.faiss import FAISS

from svsvllm.utils import CommandTimer
from svsvllm.loaders import llm_chain, load_model
from svsvllm.rag import ITALIAN_PROMPT_TEMPLATE


@pytest.mark.parametrize(
    "model_name, quantize, quantize_w_torch, model_class, tokenizer_class",
    [
        ("BEE-spoke-data/smol_llama-101M-GQA", True, True, None, None),
        # ("meta-llama/Meta-Llama-3.1-8B-Instruct", True, True), # Gated repo...
        # ("galatolo/cerbero-7b", True, True, None, None),  # Very big...
        # ("andreabac3/Fauno-Italian-LLM-7B", True, True, None, None),  # Broken on HuggingFace
        # ("Musixmatch/umberto-commoncrawl-cased-v1", True, True, None, None),
        # (
        #     "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",  # Too big...
        #     True,
        #     True,
        #     MixtralForCausalLM,
        #     LlamaTokenizer,
        # ),
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
    model_class: type[AutoModelForCausalLM] | None,
    tokenizer_class: type[AutoTokenizer] | None,
) -> None:
    """Test we can run a simple example."""
    # Load model
    model, tokenizer = load_model(
        model_name,
        bnb_config=bnb_config,
        quantize=quantize,
        quantize_w_torch=quantize_w_torch,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
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
