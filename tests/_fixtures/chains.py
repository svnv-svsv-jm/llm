import pytest
import typing as ty
from loguru import logger

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from optimum.quanto import QuantizedModelForCausalLM
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


@pytest.fixture
def tiny_llama_pipeline(
    tiny_llama: tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer],
    device: torch.device,
) -> RunnableSerializable:
    """TinyLlama LLM."""
    model, tokenizer = tiny_llama
    # Create pipeline
    pipe = HuggingFacePipeline(
        pipeline=pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map=device,
        )
    )
    # Create chain
    llm_chain = pipe | StrOutputParser()
    return llm_chain
