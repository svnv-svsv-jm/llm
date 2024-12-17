# import pytest
# from unittest.mock import patch
# from loguru import logger
# import typing as ty
# import sys, os
# import yaml

# from codetiming import Timer as CommandTimer
# import torch
# from langchain_community.vectorstores.faiss import FAISS
# from transformers import AutoTokenizer, AutoModelForCausalLM

# from svsvllm.loaders import llm_chain
# from svsvllm.rag import ITALIAN_PROMPT_TEMPLATE
# from svsvllm.defaults import DEFAULT_LLM


# @pytest.mark.parametrize("max_new_tokens", [100])
# def test_llm(
#     artifact_location: str,
#     patch_torch_quantized_engine: bool,
#     database: FAISS,
#     default_llm: tuple[AutoModelForCausalLM, AutoTokenizer],
#     device: torch.device,
#     max_new_tokens: int,
# ) -> None:
#     """Test we can run a simple example."""
#     # Load model
#     model, tokenizer = default_llm

#     # Question
#     question = "come si calcola la plusvalenza sulla cessione di criptoattività?"
#     logger.info(f"Invoking LLM with question: {question}")

#     # LLM's
#     logger.info("Creating LLMs...")
#     chain = llm_chain(
#         model,
#         tokenizer,
#         prompt_template=ITALIAN_PROMPT_TEMPLATE,
#         device=device,
#         max_new_tokens=max_new_tokens,
#     )
#     chain_w_rag = llm_chain(
#         model,
#         tokenizer,
#         database=database,
#         prompt_template=ITALIAN_PROMPT_TEMPLATE,
#         device=device,
#         max_new_tokens=max_new_tokens,
#     )

#     # Run LLM's
#     logger.info("Running LLMs...")
#     with CommandTimer(f"(no-rag)"):
#         answer_no_rag = chain.invoke({"context": "", "question": question})
#     with CommandTimer(f"(with-rag)"):
#         answer_w_rag = chain_w_rag.invoke(question)

#     # Save answers
#     logger.info("Saving LLMs' answers...")
#     name = DEFAULT_LLM.replace("/", "--")
#     answers = {name: dict(no_rag=answer_no_rag, rag=answer_w_rag)}
#     with open(os.path.join(artifact_location, f"{name}.yaml"), "w") as outfile:
#         yaml.dump(answers, outfile, indent=2)


# if __name__ == "__main__":
#     logger.remove()
#     logger.add(sys.stderr, level="TRACE")
#     pytest.main([__file__, "-x", "-s", "--pylint"])
