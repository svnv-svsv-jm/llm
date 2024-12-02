__all__ = [
    "initialize_database",
    "initialize_retriever",
    "initialize_rag",
    "create_history_aware_retriever",
]

import typing as ty
from loguru import logger
import streamlit as st
from langchain_core.retrievers import BaseRetriever, RetrieverOutputLike
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever as create_har
from langchain_community.vectorstores.faiss import FAISS

from svsvllm.rag import create_rag_database
from svsvllm.settings import settings
from .model import create_chat_model
from .session_state import SessionState


@st.cache_resource
def initialize_database(force_recreate: bool = False) -> FAISS:
    """Initialize the database.

    Args:
        force_recreate (bool, optional):
            If `True`, database is re-created even if it exists already.
            Defaults to `False`.

    Returns:
        FAISS: Created database.
    """
    state = SessionState().state
    db = state.db
    logger.trace(f"DB: {db}")
    if force_recreate or db is None:
        embedding_model_name = state.embedding_model_name
        db = create_rag_database(
            settings.uploaded_files_dir,
            model_name=embedding_model_name,
            chunk_size=state.chunk_size,
            chunk_overlap=state.chunk_overlap,
        )
        st.session_state["db"] = db
    return db


@st.cache_resource
def initialize_retriever() -> BaseRetriever:
    """Initialize the retriever."""
    state = SessionState().state
    retriever = state.retriever
    if retriever is not None:
        logger.debug(f"Retriever already initialized: {retriever}")
        return retriever
    db = state.db
    if db is None:
        logger.warning("Database not initialized, attempting to initialize it...")
        db = initialize_database()
    retriever = db.as_retriever()
    st.session_state["retriever"] = retriever
    return retriever


def initialize_rag(force_recreate: bool = False) -> BaseRetriever:
    """Initialize RAG.

    Args:
        force_recreate (bool, optional):
            If `True`, database is re-created even if it exists already.
            Defaults to `False`.

    Returns:
        BaseRetriever: Retriever.
    """
    logger.trace("Initializing RAG")
    initialize_database(force_recreate=force_recreate)
    logger.trace("Initializing retriever")
    retriever = initialize_retriever()
    return retriever


@st.cache_resource
def create_history_aware_retriever(
    force_recreate: bool = False,
    **kwargs: ty.Any,
) -> RetrieverOutputLike:
    """Create history-aware retriever.

    Args:
        force_recreate (bool, optional):
            If `True`, database is re-created even if it exists already.
            Defaults to `False`.

    Returns:
        RetrieverOutputLike:
            History aware retriever.
    """
    logger.trace("Creating history-aware retriever")
    # Get current state
    state = SessionState().state

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
    chat_model = state.chat_model
    logger.trace(f"chat_model: {chat_model}")
    if chat_model is None:
        logger.trace("Chat model not initialized, attempting to initialize it...")
        chat_model = create_chat_model(**kwargs)
        logger.trace(f"chat_model: {chat_model}")

    # RAG
    retriever = state.retriever
    logger.trace(f"retriever: {retriever}")
    if retriever is None or force_recreate:
        logger.trace("Retriever not initialized, attempting to initialize it...")
        retriever = initialize_rag(force_recreate=force_recreate)
        logger.trace(f"retriever: {retriever}")

    # Our history aware retriever
    history_aware_retriever = create_har(
        chat_model,
        retriever,
        contextualize_q_prompt,
    )
    logger.trace(f"Created history-aware retriever: {history_aware_retriever}")
    st.session_state["history_aware_retriever"] = history_aware_retriever
    return history_aware_retriever
