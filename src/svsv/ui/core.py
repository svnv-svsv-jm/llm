# pylint: disable=no-member

__all__ = ["run_ui"]

import typing as ty

import streamlit as st
from llama_index.core.llms import ChatMessage
from loguru import logger

from svsv._session_state import session_state
from svsv._settings import settings

T = ty.TypeVar("T")


def run_ui(default_response: str | None = None) -> None:
    """Run UI."""
    # Params
    default_response = default_response if default_response else settings.default_response
    logger.trace(f"Default response set to: {default_response}")

    # Title
    st.title("Chat App")

    # Accept user input
    if prompt := st.chat_input("What's up?"):
        logger.trace(f"User input: {prompt}")

        # Add user message to chat history
        session_state.messages.append(ChatMessage(content=prompt, role="user"))

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = default_response

            # Render assistant's response
            logger.trace(f"Full response: {full_response}")
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        session_state.messages.append(ChatMessage(content=full_response, role="assistant"))


if __name__ == "__main__":
    run_ui()
