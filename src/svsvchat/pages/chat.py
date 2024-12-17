# pylint: disable=no-member
__all__ = ["chat_page"]

import typing as ty
from loguru import logger
import streamlit as st
from langchain_core.messages import HumanMessage

from svsvchat.settings import settings
from svsvchat.session_state import session_state
from svsvchat.chat import initialize_chat_history, add_prompt_to_chat_history
from .sidebar import sidebar


def chat_page() -> None:
    """Main page.

    Contains the sidebar and the chat.
    """
    # Log START
    logger.debug(f"Main page: START")

    # Sidebar
    if settings.has_sidebar:
        sidebar()

    # Title and initialization
    logger.trace("Title and initialization")
    st.title(settings.app_title)
    st.subheader(settings.app_subheader)
    st.caption(settings.app_caption)

    # Initialize messages if not done yet
    initialize_chat_history()
    logger.trace("Initialized messages")

    # Chat?
    has_chat: bool = settings.has_chat
    logger.trace(f"Has chat: {has_chat}")
    if not has_chat:
        session_state.chat_activated = False
        return

    # Get user's prompt
    logger.trace("Starting chat loop")
    prompt = st.chat_input()

    # If prompt exists, add it to chat history
    add_prompt_to_chat_history(prompt)

    # Log END
    logger.debug(f"Main page: END")
