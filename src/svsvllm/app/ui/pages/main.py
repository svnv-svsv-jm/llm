__all__ = ["main_page"]

import typing as ty
from loguru import logger
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from ..response import get_openai_response, get_response_from_open_source_model
from ..sidebar import sidebar
from ..defaults import DEFAULT_MODEL
from ..messages import initialize_messages


def main_page() -> None:
    """Main page.

    Contains the sidebar and the chat.
    """
    # Sidebar
    sidebar()

    # Title and initialization
    st.title("💬 FiscalAI")
    st.caption("🚀 Your favorite chatbot, powered by FiscalAI.")

    # Initialize messages if not done yet
    initialize_messages()
    logger.trace("Initialized messages")

    # Main chat loop
    logger.trace("Starting chat loop")
    if prompt := st.chat_input():
        # Start session
        logger.trace(f"Received prompt: {prompt}")
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))

        # LLM's response
        openai_api_key = st.session_state.get("openai_api_key", None)
        with st.chat_message("assistant"):
            if openai_api_key is not None:
                msg = get_openai_response(
                    model=st.session_state.get("model", DEFAULT_MODEL),
                    openai_api_key=st.session_state.get("openai_api_key", None),
                )
                st.write(msg)
                message = AIMessage(content=msg)
            else:
                response = st.write_stream(get_response_from_open_source_model(prompt))
                message = AIMessage(content=response)

            # Update session
            logger.trace(f"Assistant: {msg}")
            st.session_state.messages.append(message)
