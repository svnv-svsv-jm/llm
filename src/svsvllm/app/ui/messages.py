__all__ = ["initialize_messages"]

from loguru import logger
import streamlit as st
from langchain_core.messages import AIMessage

from .start_messages import START_MSG_EN
from .session_state import SessionState


def initialize_messages() -> None:
    """Initialize messages if not done yet."""
    session_state = SessionState()
    if not session_state.state.messages:
        init_message = AIMessage(content=START_MSG_EN)
        logger.trace(f"Initializing messages: {init_message}")
        with st.chat_message("assistant"):
            st.write(init_message.content)
        session_state["messages"] = [init_message]
