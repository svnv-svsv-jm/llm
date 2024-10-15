__all__ = ["tiny_llama_chain"]

import pytest
import typing as ty
from loguru import logger

from transformers import Pipeline
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline


@pytest.fixture
def tiny_llama_chain(tiny_llama_pipeline: Pipeline) -> RunnableSerializable:
    """TinyLlama LLM."""
    # Create pipeline
    pipe = HuggingFacePipeline(pipeline=tiny_llama_pipeline)
    # Create chain
    llm_chain = pipe | StrOutputParser()
    return llm_chain
