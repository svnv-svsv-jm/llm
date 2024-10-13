__all__ = ["initialize_rag"]

import streamlit as st

from svsvllm.rag import create_rag_database
from .const import UPLOADED_FILES_DIR


def initialize_rag(force_recreate: bool = False) -> None:
    """Initialize RAG."""
    if force_recreate or "db" not in st.session_state:
        db = create_rag_database(UPLOADED_FILES_DIR)
        retriever = db.as_retriever()
        st.session_state["retriever"] = retriever
        st.session_state["db"] = db
