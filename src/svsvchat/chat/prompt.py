__all__ = ["add_prompt_to_chat_history"]

from loguru import logger
from langchain_core.messages import HumanMessage
import streamlit as st

from svsvchat.session_state import session_state


def add_prompt_to_chat_history(prompt: str | None) -> None:
    """Add prompt to chat history.

    Args:
        prompt (str | None):
            User's prompt / message.
    """
    # If prompt does not exist, do nothing
    if not prompt:
        return

    # If prompt exists, add it to chat history
    # Create message
    logger.trace(f"Received prompt: '{prompt}'")
    message = HumanMessage(content=prompt)

    # Add prompt to session
    session_state.chat_activated = True
    with st.chat_message(message.type):
        logger.trace(f"Calling `st.markdown('{prompt}')`")
        st.markdown(prompt)
    session_state.chat_history.append(message)  # pylint: disable=no-member
