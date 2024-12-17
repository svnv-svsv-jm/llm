__all__ = ["initialize_chat_history"]

from loguru import logger
import streamlit as st
from langchain_core.messages import AIMessage

from svsvllm.settings import settings
from .session_state import session_state


def initialize_chat_history() -> None:
    """Initialize messages if not done yet."""
    logger.trace(f"Chat history: {session_state.chat_history}")
    init_message = AIMessage(content=settings.start_message_en)

    if len(session_state.chat_history) == 0:
        logger.trace(f"Initializing chat history: {init_message}")
        session_state.chat_history = [init_message]

    with st.chat_message("assistant"):
        st.write(init_message.content)
