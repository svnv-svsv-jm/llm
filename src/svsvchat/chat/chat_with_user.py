__all__ = ["chat_with_user"]

from loguru import logger
import streamlit as st
from langchain_core.messages import AIMessage

from svsvchat.session_state import session_state
from .get_openai_response import get_openai_response
from .stream_open_source_model import stream_open_source_model


def chat_with_user(prompt: str | None = None) -> AIMessage:
    """Chat with user."""
    # If no prompt, return
    if not prompt:
        return

    # LLM's response
    with st.chat_message("assistant"):
        # Check if we have to run OpenAI
        openai_api_key = session_state.openai_api_key
        logger.debug(f"Received OpenAI API key: {openai_api_key}")
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
            response = st.write_stream(stream_open_source_model(prompt))
            logger.trace(f"Creating `AIMessage`: {response}")
            message = AIMessage(content=response)

        # Update session
        logger.trace(f"Assistant: {message}")
        session_state.chat_history.append(message)  # pylint: disable=no-member

    return message
