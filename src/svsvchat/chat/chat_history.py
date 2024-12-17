__all__ = ["initialize_chat_history"]

from loguru import logger
import streamlit as st
from langchain_core.messages import AIMessage

from svsvchat.session_state import session_state
from .initial_message import select_init_msg


def initialize_chat_history() -> None:
    """Initialize messages if not done yet."""
    logger.trace(f"Chat history: {session_state.chat_history}")

    if len(session_state.chat_history) == 0:
        logger.trace("Initializing chat history")
        init_message = AIMessage(content=select_init_msg())
        logger.trace(f"Initial message: {init_message}")
        session_state.chat_history = [init_message]

    # Display previous messages
    logger.trace("Displaying previous messages")
    for message in session_state.chat_history:
        with st.chat_message(message.type):
            content = message.content
            if not isinstance(content, list):
                content = [content]
            for c in content:
                st.markdown(c)
