__all__ = ["get_response"]

import streamlit as st
from openai import OpenAI

from .defaults import DEFAULT_MODEL


def get_openai_response(openai_api_key: str, model: str) -> str | None:
    """Returns the response from the OpenAI model.
    The full chat history up to now is fed as input.
    """
    # OpenAI
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=st.session_state.messages,
    )
    msg = response.choices[0].message.content
    return msg


def get_response_from_open_source_model() -> str:
    """Not supported yet."""
    # TODO: initialize RAG, initialize model, create tools, create agent.

    # Inform user via warning box
    st.info(
        "Add your OpenAI API key to continue. Support for open-source models is in development."
    )
    # Let the chatbox also inform the user
    return "Welcome to FiscalAI! Unfortunately, support for open-source models is still in development. Please add your OpenAI API key to get a different, meaningful response."


def get_response(
    model: str = DEFAULT_MODEL,
    openai_api_key: str | None = None,
) -> str:
    """Get response from the chatbot.

    Args:
        model (str | None, optional):
            Model ID or name.
            Defaults to `"gpt-3.5-turbo"`.

        openai_api_key (str | None, optional):
            OpenAI key.
            Defaults to `None`.

    Returns:
        str: response from the LLM.
    """
    # No OpenAI
    if not openai_api_key:
        return get_response_from_open_source_model()

    # OpenAI
    msg = get_openai_response(openai_api_key, model)

    # Sanity check and return
    if msg is None:
        msg = ""
    return f"{msg}"
