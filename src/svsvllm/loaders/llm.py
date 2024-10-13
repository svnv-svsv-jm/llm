__all__ = ["llm_chain"]

import typing as ty
from loguru import logger
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from optimum.quanto import QuantizedModelForCausalLM

from svsvllm.rag import DEFAULT_TEMPLATE


def llm_chain(
    model: AutoModelForCausalLM | QuantizedModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_template: str = DEFAULT_TEMPLATE,
    database: FAISS | None = None,
    task: str = "text-generation",
    **kwargs: ty.Any,
) -> Runnable:
    """LLM chain.

    Args:
        model (AutoModelForCausalLM | QuantizedModelForCausalLM):
            LLM model for the chain.

        tokenizer (AutoTokenizer):
            Tokenizer.

        prompt_template (str, optional):
            Prompt template. Defaults to `DEFAULT_TEMPLATE`.

        database (FAISS | None, optional):
            Database for the RAG. Defaults to `None`.
            If no database, then no RAG.

        task (str):
            The task defining which pipeline will be returned.
            See `transformers.pipeline`.
            Defaults to `"text-generation"`.

    Returns:
        RunnableSerializable: LLM model, with or without RAG.
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    pipe = HuggingFacePipeline(
        pipeline=pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )
    )
    llm = prompt | pipe | StrOutputParser()
    logger.debug(f"LLM chain: {llm}")
    if database is not None:
        retriever = database.as_retriever()
        llm = {
            "context": retriever,
            "question": RunnablePassthrough(),
        } | llm
    return llm
