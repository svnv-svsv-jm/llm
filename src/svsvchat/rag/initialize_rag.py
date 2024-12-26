__all__ = ["initialize_database", "initialize_retriever", "initialize_rag"]

from loguru import logger
import streamlit as st
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.retrievers import BaseRetriever

from svsvchat.session_state import session_state
from svsvchat.settings import settings
from svsvllm.rag import create_rag_database
from .history_aware_retriever import create_history_aware_retriever


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
    logger.trace("Initializing database")
    db = session_state.db
    logger.trace(f"DB: {db}")
    if force_recreate or db is None:
        embedding_model_name = session_state.embedding_model_name
        db = create_rag_database(
            settings.uploaded_files_dir,
            model_name=embedding_model_name,
            chunk_size=session_state.chunk_size,
            chunk_overlap=session_state.chunk_overlap,
        )
        st.session_state["db"] = db
        session_state.manual_sync("db", reverse=True)
    logger.trace(f"Initialized database: {db}")
    return db


@st.cache_resource
def initialize_retriever() -> BaseRetriever:
    """Initializes the retriever.

    Returns:
        BaseRetriever:
            Database document retriver.
    """
    retriever = session_state.retriever
    if retriever is not None:
        logger.debug(f"Retriever already initialized: {retriever}")
        return retriever
    db = session_state.db
    if db is None:
        logger.warning("Database not initialized, attempting to initialize it...")
        db = initialize_database()
    retriever = db.as_retriever()
    st.session_state["retriever"] = retriever
    session_state.manual_sync("retriever", reverse=True)
    return retriever


def initialize_rag(force_recreate: bool = False) -> tuple[FAISS, BaseRetriever]:
    """Initializes the RAG.

    Args:
        force_recreate (bool, optional):
            If `True`, RAG is recreated even if it existed already.
            Defaults to `False`.

    Returns:
        tuple[FAISS, BaseRetriever]:
            The document database and the document chunk retriever.
    """
    logger.trace("Initializing RAG")
    db = initialize_database(force_recreate=force_recreate)
    logger.trace("RAG initialized")
    logger.trace("Initializing retriever")
    retriever = initialize_retriever()
    logger.trace("Retriever initialized ")
    logger.trace("Initializing history-aware retriever")
    create_history_aware_retriever()
    logger.trace("History-aware retriever initialized")
    return db, retriever
