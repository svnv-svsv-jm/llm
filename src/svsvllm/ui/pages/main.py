# pylint: disable=no-member
__all__ = ["main_page"]

import typing as ty
from loguru import logger
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from svsvllm.settings import settings
from ..response import get_openai_response, stream_open_source_model, invoke_open_source_model
from ..messages import initialize_chat_history
from ..session_state import session_state
from .sidebar import sidebar


def main_page() -> None:
    """Main page.

    Contains the sidebar and the chat.
    """
    # Log START
    logger.debug(f"Main page: START")

    # Manually sync stuff
    session_state.manual_sync("chat_activated")

    # Sidebar
    if settings.has_sidebar:
        sidebar()

    # Title and initialization
    logger.debug("Title and initialization")
    st.title(settings.app_title)
    st.subheader(settings.app_subheader)
    st.caption(settings.app_caption)

    # Initialize messages if not done yet
    initialize_chat_history()
    logger.debug("Initialized messages")
    messages = session_state.chat_history

    # Chat?
    has_chat: bool = settings.has_chat
    logger.debug(f"Has chat: {has_chat}")
    if not has_chat:
        session_state.chat_activated = False
        return

    # Main chat loop
    logger.trace("Starting chat loop")
    if prompt := st.chat_input():
        # Add prompt to session
        session_state.chat_activated = True
        logger.trace(f"Received prompt: {prompt}")
        with st.chat_message("user"):
            st.markdown(prompt)
        messages.append(HumanMessage(content=prompt))

        # LLM's response
        with st.chat_message("assistant"):
            # Check if we have to run OpenAI
            openai_api_key = session_state.openai_api_key
            logger.debug(f"Received OpenAI API key: {type(openai_api_key)}")
            if openai_api_key is not None:
                logger.trace("Calling OpenAI model")
                msg = get_openai_response(
                    model=session_state.openai_model_name,
                    openai_api_key=str(openai_api_key),
                )
                st.write(msg)
                message = AIMessage(content=msg)

            # Run open source model?
            else:
                logger.trace("Calling open source model")
                if session_state.streaming:
                    response = st.write_stream(stream_open_source_model(prompt))
                else:
                    response = invoke_open_source_model(prompt)
                    st.write(response)
                logger.trace("Creating `AIMessage`")
                message = AIMessage(content=response)

            # Update session
            logger.trace(f"Assistant: {message}")
            messages.append(message)
            session_state.messages = messages  # This reruns validation

    # Log END
    logger.debug(f"Main page: END")
