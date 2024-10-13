__all__ = ["main_page"]

from loguru import logger
import streamlit as st

from ..response import get_response
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
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # LLM's response
        msg = get_response(
            model=st.session_state.get("model", DEFAULT_MODEL),
            openai_api_key=st.session_state.get("openai_api_key", None),
        )

        # Update session
        logger.trace(f"Assistant: {msg}")
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
