__all__ = ["get_response"]

import streamlit as st
from openai import OpenAI

from .defaults import DEFAULT_MODEL


def get_response(
    model: str | None = None,
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
    # Parse inputs
    if model is None:
        model = DEFAULT_MODEL

    # No OpenAI
    if not openai_api_key:
        st.info(
            "Add your OpenAI API key to continue. Support for open-source models is in development."
        )
        return "Welcome to FiscalAI! Unfortunately, support for open-source models is still in development. Please add your OpenAI API key to get a different, meaningful response."

    # OpenAI
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=st.session_state.messages,
    )
    msg = response.choices[0].message.content
    if msg is None:
        msg = ""
    return msg
