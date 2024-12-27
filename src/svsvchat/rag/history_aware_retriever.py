__all__ = ["create_history_aware_retriever"]

import typing as ty
from loguru import logger
import streamlit as st
from langchain_core.retrievers import RetrieverOutputLike
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever as create_har

from svsvllm.exceptions import RetrieverNotInitializedError
from svsvchat.session_state import session_state
from svsvchat.settings import settings
from svsvchat.model import create_chat_model


@st.cache_resource
def create_history_aware_retriever(**kwargs: ty.Any) -> RetrieverOutputLike:
    """Create history-aware retriever.

    Args:
        force_recreate (bool, optional):
            If `True`, database is re-created even if it exists already.
            Defaults to `False`.

        **kwargs:
            See :func:`create_chat_model`.

    Returns:
        RetrieverOutputLike:
            History aware retriever.
    """
    logger.trace("Creating history-aware retriever")
    # Get current state
    state = session_state

    # RAG
    logger.trace(f"retriever: {state.retriever}")
    if state.retriever is None:
        raise RetrieverNotInitializedError()

    # Create prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", settings.q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    logger.trace(f"contextualize_q_prompt: {contextualize_q_prompt}")

    # Chat model
    logger.trace(f"chat_model: {state.chat_model}")
    if state.chat_model is None:
        logger.trace("Chat model not initialized, attempting to initialize it...")
        chat_model = create_chat_model(**kwargs)
        logger.trace(f"chat_model: {chat_model}")

    # Our history aware retriever
    history_aware_retriever = create_har(chat_model, state.retriever, contextualize_q_prompt)
    logger.trace(f"Created history-aware retriever: {history_aware_retriever}")
    state.history_aware_retriever = history_aware_retriever
    st.session_state["history_aware_retriever"] = history_aware_retriever
    session_state.manual_sync("history_aware_retriever", reverse=True)
    return history_aware_retriever
