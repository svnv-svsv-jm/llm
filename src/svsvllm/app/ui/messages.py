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
        st.session_state["messages"] = [init_message]

    # for msg in st.session_state.messages:
    #     logger.trace(f"Writing message: `{msg}`.")
    #     assert isinstance(msg, BaseMessage)
    #     st.chat_message(role).write(msg.content)
