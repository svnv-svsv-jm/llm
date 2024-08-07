import pytest
from loguru import logger
import typing as ty
import sys, os
import yaml

from langchain_core.runnables import RunnableSerializable

from svsvllm.utils import CommandTimer


def test_llm_cerbero(
    llm_chain_cerbero: RunnableSerializable,
    llm_chain_cerbero_w_rag: RunnableSerializable,
    artifact_location: str,
) -> None:
    """Test we can run a simple example."""
    # Question
    question = "come si calcola la plusvalenza sulla cessione di criptoattività?"
    logger.info(f"Invoking LLM with question: {question}")
    # Run LLM's
    with CommandTimer():
        answer_no_rag = llm_chain_cerbero.invoke({"context": "", "question": question})
    with CommandTimer():
        answer_w_rag = llm_chain_cerbero_w_rag.invoke(question)
    answers = dict(cerbero=dict(no_rag=answer_no_rag, rag=answer_w_rag))
    # Save
    with open(os.path.join(artifact_location, "cerbero.yaml"), "w") as outfile:
        yaml.dump(answers, outfile, indent=2)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
