__all__ = ["initialize_rag"]

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from svsvllm.rag import create_rag_database
from .const import UPLOADED_FILES_DIR, Q_SYSTEM_PROMPT


def initialize_rag(force_recreate: bool = False) -> None:
    """Initialize RAG."""
    if force_recreate or "db" not in st.session_state:
        db = create_rag_database(UPLOADED_FILES_DIR)
        retriever = db.as_retriever()
        st.session_state["retriever"] = retriever
        st.session_state["db"] = db


def create_retriever() -> None:
    """Create history-aware retriever."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        st.session_state.model,
        st.session_state.retriever,
        contextualize_q_prompt,
    )
    st.session_state["history_aware_retriever"] = history_aware_retriever
