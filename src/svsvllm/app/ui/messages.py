__all__ = ["initialize_messages"]

from loguru import logger
import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage

from .start_messages import START_MSG_EN


def initialize_messages() -> None:
    """Initialize messages if not done yet."""
    if "messages" not in st.session_state:
        init_message = AIMessage(content=START_MSG_EN)
        logger.trace(f"Initializing messages: {init_message}")
        with st.chat_message("assistant"):
            st.write(init_message.content)
        st.session_state["messages"] = [init_message]
