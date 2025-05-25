# pylint: disable=no-member

__all__ = ["run_ui"]

import typing as ty

import streamlit as st
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.llms.ollama import Ollama
from loguru import logger

from svsv._session_state import session_state
from svsv._settings import settings

T = ty.TypeVar("T")


@st.cache_resource
def load_llm(model: str | None = None) -> FunctionCallingLLM:
    """`Ollama`."""
    model = settings.default_llm if model is None else model
    llm = Ollama(model, request_timeout=120.0)
    return llm


def set_up_llm() -> FunctionCallingLLM:
    """Set up LLM."""
    logger.trace("Setting up LLM...")
    llm = load_llm()
    Settings.llm = llm
    session_state.llm = llm
    logger.trace(f"Set up LLM: {llm}")
    return llm


def run_ui(default_response: str | None = None) -> None:
    """Run UI."""
    # Params
    default_response = default_response if default_response else settings.default_response
    logger.trace(f"Default response set to: {default_response}")

    # Set up LLM
    llm = set_up_llm()
    logger.trace(f"Set up: {llm}")

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

            # Get response from LLM
            streamer = llm.stream_chat(session_state.messages)
            for chunk in streamer:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            # Render assistant's response
            logger.trace(f"Full response: {full_response}")
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        session_state.messages.append(ChatMessage(content=full_response, role="assistant"))


if __name__ == "__main__":
    run_ui()
