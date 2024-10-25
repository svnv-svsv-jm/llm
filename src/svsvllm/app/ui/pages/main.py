__all__ = ["main_page"]

import typing as ty
from loguru import logger
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from svsvllm.app.const import OPEN_SOURCE_MODELS_SUPPORTED
from svsvllm.app.settings import settings
from ..response import get_openai_response, get_response_from_open_source_model
from ..messages import initialize_messages
from ..session_state import SessionState
from .sidebar import sidebar


def main_page() -> None:
    """Main page.

    Contains the sidebar and the chat.
    """
    # Log START
    logger.debug(f"Main page: START")

    # Get current state
    state = SessionState().state

    # Manually sync stuff
    SessionState().manual_sync("chat_activated")

    # Sidebar
    if settings.has_sidebar:
        sidebar()

    # Title and initialization
    logger.debug("Title and initialization")
    st.title(settings.app_title)
    st.subheader(settings.app_subheader)
    st.caption(settings.app_caption)

    # Initialize messages if not done yet
    initialize_messages()
    logger.debug("Initialized messages")
    messages = state.messages

    # Chat?
    has_chat: bool = settings.has_chat
    logger.debug(f"Has chat: {has_chat}")
    if not has_chat:
        state.chat_activated = False
        return

    # Main chat loop
    logger.trace("Starting chat loop")
    if prompt := st.chat_input():
        # Start session
        state.chat_activated = True
        logger.trace(f"Received prompt: {prompt}")
        with st.chat_message("user"):
            st.markdown(prompt)
        messages.append(HumanMessage(content=prompt))

        # LLM's response
        with st.chat_message("assistant"):
            # Check if we have to run OpenAI
            openai_api_key = state.openai_api_key
            logger.debug(f"Received OpenAI API key: {type(openai_api_key)}")
            if openai_api_key is not None:
                logger.trace("Calling OpenAI model")
                msg = get_openai_response(
                    model=state.openai_model_name,
                    openai_api_key=str(openai_api_key),
                )
                st.write(msg)
                message = AIMessage(content=msg)

            # Run open source model?
            else:
                if OPEN_SOURCE_MODELS_SUPPORTED:
                    logger.trace("Calling open source model")
                    response = st.write_stream(get_response_from_open_source_model(prompt))
                else:
                    logger.trace("Open source models not supported message")
                    # Let the chatbox inform the user
                    response = "Welcome to FiscalAI! Unfortunately, support for open-source models is still in development. Please add your OpenAI API key to get a different, meaningful response."  # pragma: no cover
                    st.write(response)  # pragma: no cover
                logger.trace("Creating `AIMessage`")
                message = AIMessage(content=response)

            # Update session
            logger.trace(f"Assistant: {message}")
            messages.append(message)
            state.messages = messages  # This reruns validation

    # Log END
    logger.debug(f"Main page: END")
