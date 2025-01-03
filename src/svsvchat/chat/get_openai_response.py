__all__ = ["get_openai_response"]

import typing as ty
from loguru import logger
import streamlit as st
from openai import OpenAI

from svsvchat.session_state import session_state


def get_openai_response(openai_api_key: str, model: str) -> str:
    """Returns the response from the OpenAI model.

    The full chat history up to now is fed as input.

    Args:
        openai_api_key (str):
            OpenAI API key.

        model (str):
            OpenAI model name.

    Returns:
        str: LLM's response as text.
    """
    state = session_state
    # OpenAI client
    if "openai_client" not in st.session_state:
        logger.trace(f"Creating new OpenAI client.")
        client = OpenAI(api_key=openai_api_key)
        st.session_state["openai_client"] = client
        logger.trace(f"Created new OpenAI client.")
    else:
        logger.trace(f"Getting OpenAI client from session state.")
        client = st.session_state["openai_client"]
        logger.trace(f"Got OpenAI client from session state.")

    # Create response
    logger.trace(f"Calling `OpenAI.chat.completions.create`")
    response = client.chat.completions.create(
        model=model,
        # NOTE: This param is type-ignored because even though the type is wrong, all that matters is that the provided objects (in the list) implement the required class properties: content, role, name
        messages=state.chat_history,  # type: ignore
    )
    msg = response.choices[0].message.content

    # Sanity check and return
    if msg is None:
        msg = ""
    return f"{msg}"
