__all__ = ["main_page"]

import typing as ty
from loguru import logger
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from svsvllm.defaults import OPENAI_DEFAULT_MODEL
from ..response import get_openai_response, get_response_from_open_source_model
from ..sidebar import sidebar
from ..messages import initialize_messages
from ..const import OPEN_SOURCE_MODELS_SUPPORTED


def main_page() -> None:
    """Main page.

    Contains the sidebar and the chat.
    """
    # Sidebar
    sidebar()

    # Title and initialization
    st.title("💬 FiscalAI")
    st.subheader("Smart Assistant")
    st.caption("🚀 Your favorite chatbot, powered by FiscalAI.")

    # Initialize messages if not done yet
    initialize_messages()
    logger.trace("Initialized messages")
    messages: list[BaseMessage] = st.session_state.messages

    # Chat?
    has_chat: bool = st.session_state.get("has_chat", True)
    logger.trace(f"Has chat: {has_chat}")
    if not has_chat:
        return

    # Main chat loop
    logger.trace("Starting chat loop")
    prompt = st.chat_input()  # Prompt from user
    if prompt:
        # Start session
        logger.trace(f"Received prompt: {prompt}")
        with st.chat_message("user"):
            st.markdown(prompt)
        messages.append(HumanMessage(content=prompt))

        # LLM's response
        with st.chat_message("assistant"):
            # Check if we have to run OpenAI
            openai_api_key: str | None = st.session_state.get("openai_api_key", None)
            logger.trace(f"Received OpenAI API key: {type(openai_api_key)}")
            if openai_api_key is not None:
                logger.trace("Calling OpenAI model")
                msg = get_openai_response(
                    model=st.session_state.get("model", OPENAI_DEFAULT_MODEL),
                    openai_api_key=openai_api_key,
                )
                st.write(msg)
                message = AIMessage(content=msg)

            # Run open source model?
            else:
                if OPEN_SOURCE_MODELS_SUPPORTED:
                    logger.trace("Calling open source model")
                    response = st.write_stream(get_response_from_open_source_model(prompt))
                else:
                    logger.trace("Ppen source models not supported message")
                    # Let the chatbox inform the user
                    response = "Welcome to FiscalAI! Unfortunately, support for open-source models is still in development. Please add your OpenAI API key to get a different, meaningful response."  # pragma: no cover
                    st.write(response)  # pragma: no cover
                logger.trace("Creating `AIMessage`")
                message = AIMessage(content=response)

            # Update session
            logger.trace(f"Assistant: {message}")
            messages.append(message)