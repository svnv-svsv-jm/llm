# pylint: disable=no-member

__all__ = ["run_ui"]

import typing as ty

import streamlit as st
from llama_index.core.llms import ChatMessage
from loguru import logger

from svsv._session_state import session_state

T = ty.TypeVar("T")


def run_ui() -> None:
    """Run UI."""
    # Title
    st.title("Chat App")

    # Accept user input
    if prompt := st.chat_input("What's up?"):
        logger.trace(f"User input: {prompt}")

        # Add user message to chat history
        session_state.messages.append(ChatMessage(**{"role": "user", "content": prompt}))

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)


if __name__ == "__main__":
    run_ui()
