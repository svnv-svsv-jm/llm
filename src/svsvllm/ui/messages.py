__all__ = ["initialize_messages"]

from loguru import logger
import streamlit as st
from langchain_core.messages import AIMessage

from svsvllm.settings import settings
from .session_state import SessionState


def initialize_messages() -> None:
    """Initialize messages if not done yet."""
    session_state = SessionState()
    logger.trace(f"Initializing messages: {session_state.state.messages}")
    init_message = AIMessage(content=settings.start_message_en)
    if len(session_state.state.messages) == 0:
        logger.trace(f"Initializing messages: {init_message}")
        session_state["messages"] = [init_message]
    with st.chat_message("assistant"):
        st.write(init_message.content)
