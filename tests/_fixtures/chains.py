__all__ = ["llm_chain"]

import pytest
import typing as ty
from loguru import logger

from transformers import Pipeline
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline


@pytest.fixture
def llm_chain(llm_pipeline: Pipeline) -> RunnableSerializable:
    """Default LLM."""
    # Create pipeline
    pipe = HuggingFacePipeline(pipeline=llm_pipeline)
    # Create chain
    llm_chain = pipe | StrOutputParser()
    return llm_chain
