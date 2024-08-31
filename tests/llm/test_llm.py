import pytest
from loguru import logger
import typing as ty
import sys, os
import yaml

from transformers import BitsAndBytesConfig
from langchain_community.vectorstores.faiss import FAISS

from svsvllm.utils import CommandTimer
from svsvllm.loaders import llm_chain, load_model
from svsvllm.rag import ITALIAN_PROMPT_TEMPLATE


@pytest.mark.parametrize(
    "model_name, quantize, quantize_w_torch",
    [
        ("TinyLlama/TinyLlama_v1.1", True, True),
        ("BEE-spoke-data/smol_llama-101M-GQA", True, False),
    ],
)
def test_llm(
    model_name: str,
    artifact_location: str,
    database: FAISS,
    bnb_config: BitsAndBytesConfig,
    quantize: bool,
    quantize_w_torch: bool,
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
    llm = llm_chain(model, tokenizer, template=ITALIAN_PROMPT_TEMPLATE)
    llm_w_rag = llm_chain(model, tokenizer, database=database, template=ITALIAN_PROMPT_TEMPLATE)
    # Run LLM's
    with CommandTimer(f"{model_name} (no-rag)"):
        answer_no_rag = llm.invoke({"context": "", "question": question})
    with CommandTimer(f"{model_name} (with-rag)"):
        answer_w_rag = llm_w_rag.invoke(question)
    answers = dict(cerbero=dict(no_rag=answer_no_rag, rag=answer_w_rag))
    # Save
    with open(os.path.join(artifact_location, "cerbero.yaml"), "w") as outfile:
        yaml.dump(answers, outfile, indent=2)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
