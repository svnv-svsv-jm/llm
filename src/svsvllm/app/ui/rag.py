__all__ = [
    "initialize_database",
    "initialize_retriever",
    "initialize_rag",
    "create_history_aware_retriever",
]

from loguru import logger
import streamlit as st
from langchain_core.retrievers import BaseRetriever, RetrieverOutputLike
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever as create_har
from langchain_community.vectorstores.faiss import FAISS

from svsvllm.rag import create_rag_database
from svsvllm.app.settings import settings
from .model import create_chat_model
from .session_state import SessionState


@st.cache_resource
def initialize_database(force_recreate: bool = False) -> FAISS:
    """Initialize the database."""
    session_state = SessionState()
    db = session_state.state.db
    if force_recreate or db is None:
        embedding_model_name = session_state.state.embedding_model_name
        db = create_rag_database(
            settings.uploaded_files_dir,
            model_name=embedding_model_name,
            chunk_size=session_state.state.chunk_size,
            chunk_overlap=session_state.state.chunk_overlap,
        )
        st.session_state["db"] = db
    return db


@st.cache_resource
def initialize_retriever() -> BaseRetriever:
    """Initialize the retriever."""
    retriever: BaseRetriever = st.session_state.get("retriever", None)
    if retriever is not None:
        logger.debug(f"Retriever already initialized: {retriever}")
        return retriever
    session_state = SessionState()
    db: FAISS = session_state.state.db
    if db is None:
        logger.warning("Database not initialized, attempting to initialize it...")
        initialize_database()
    retriever = db.as_retriever()
    st.session_state["retriever"] = retriever
    return retriever


def initialize_rag(force_recreate: bool = False) -> BaseRetriever:
    """Initialize RAG."""
    initialize_database(force_recreate=force_recreate)
    retriever = initialize_retriever()
    return retriever


@st.cache_resource
def create_history_aware_retriever(force_recreate: bool = False) -> RetrieverOutputLike:
    """Create history-aware retriever."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", settings.q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chat_model = st.session_state.get("chat_model", None)
    if chat_model is None:
        logger.warning("Chat model not initialized, attempting to initialize it...")
        create_chat_model()
    session_state = SessionState()
    retriever = session_state.state.retriever
    if retriever is None or force_recreate:
        logger.warning("Retriever not initialized, attempting to initialize it...")
        retriever = initialize_rag(force_recreate=force_recreate)
    history_aware_retriever = create_har(
        chat_model,
        retriever,
        contextualize_q_prompt,
    )
    st.session_state["history_aware_retriever"] = history_aware_retriever
    return history_aware_retriever
