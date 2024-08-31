__all__ = ["llm_chain"]

import typing as ty
from loguru import logger
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable, RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.quanto import QuantizedModelForCausalLM

from svsvllm.rag import DEFAULT_TEMPLATE
from .pipeline import pipeline


def llm_chain(
    model: AutoModelForCausalLM | QuantizedModelForCausalLM,
    tokenizer: AutoTokenizer,
    template: str = DEFAULT_TEMPLATE,
    database: FAISS | None = None,
    **kwargs: ty.Any,
) -> RunnableSerializable:
    """LLM chain."""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    llm = prompt | pipeline(model=model, tokenizer=tokenizer, **kwargs) | StrOutputParser()
    logger.debug(f"LLM chain: {llm}")
    if database is not None:
        retriever = database.as_retriever()
        llm = {
            "context": retriever,
            "question": RunnablePassthrough(),
        } | llm
    return llm
