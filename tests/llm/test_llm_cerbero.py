import pytest
from loguru import logger
import typing as ty
import sys, os
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.quanto import QuantizedModelForCausalLM
from langchain_community.vectorstores.faiss import FAISS

from svsvllm.utils import CommandTimer
from svsvllm.loaders import llm_chain
from svsvllm.rag import ITALIAN_PROMPT_TEMPLATE


def test_llm_cerbero(
    cerbero: ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer],
    artifact_location: str,
    database: FAISS,
) -> None:
    """Test we can run a simple example."""
    model, tokenizer = cerbero
    # Question
    question = "come si calcola la plusvalenza sulla cessione di criptoattività?"
    logger.info(f"Invoking LLM with question: {question}")
    # LLM's
    llm = llm_chain(model, tokenizer, template=ITALIAN_PROMPT_TEMPLATE)
    llm_w_rag = llm_chain(model, tokenizer, database=database, template=ITALIAN_PROMPT_TEMPLATE)
    # Run LLM's
    with CommandTimer():
        answer_no_rag = llm.invoke({"context": "", "question": question})
    with CommandTimer():
        answer_w_rag = llm_w_rag.invoke(question)
    answers = dict(cerbero=dict(no_rag=answer_no_rag, rag=answer_w_rag))
    # Save
    with open(os.path.join(artifact_location, "cerbero.yaml"), "w") as outfile:
        yaml.dump(answers, outfile, indent=2)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
