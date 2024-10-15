__all__ = [
    "initialize_database",
    "initialize_retriever",
    "initialize_rag",
    "create_history_aware_retriever",
]

from loguru import logger
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever as create_har
from langchain_community.vectorstores.faiss import FAISS

from svsvllm.rag import create_rag_database
from .const import UPLOADED_FILES_DIR, Q_SYSTEM_PROMPT
from .model import create_chat_model


def initialize_database(force_recreate: bool = False) -> None:
    """Initialize the database."""
    if force_recreate or "db" not in st.session_state:
        db = create_rag_database(UPLOADED_FILES_DIR)
        st.session_state["db"] = db


def initialize_retriever() -> None:
    """Initialize the retriever."""
    retriever = st.session_state.get("retriever", None)
    if retriever is not None:
        logger.debug(f"Retriever already initialized: {retriever}")
        return
    db: FAISS = st.session_state.get("db", None)
    if db is None:
        logger.warning("Database not initialized, attempting to initialize it...")
        initialize_database()
    retriever = db.as_retriever()
    st.session_state["retriever"] = retriever


def initialize_rag(force_recreate: bool = False) -> None:
    """Initialize RAG."""
    initialize_database(force_recreate=force_recreate)
    initialize_retriever()


def create_history_aware_retriever() -> None:
    """Create history-aware retriever."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    chat_model = st.session_state.get("chat_model", None)
    if chat_model is None:
        logger.warning("Chat model not initialized, attempting to initialize it...")
        create_chat_model()
    retriever = st.session_state.get("retriever", None)
    if retriever is None:
        logger.warning("Retriever not initialized, attempting to initialize it...")
        initialize_retriever()
    history_aware_retriever = create_har(
        chat_model,
        retriever,
        contextualize_q_prompt,
    )
    st.session_state["history_aware_retriever"] = history_aware_retriever
