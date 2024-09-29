__all__ = ["get_response"]

import streamlit as st
from openai import OpenAI


def get_response(openai_api_key: str | None = None) -> str:
    """Get response from the chatbot."""
    # No OpenAI
    if not openai_api_key:
        st.info(
            "Add your OpenAI API key to continue. Support for open-source models is in development."
        )
        return "Hi, welcome to FiscalAI!"

    # OpenAI
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages,
    )
    msg = response.choices[0].message.content
    if msg is None:
        msg = ""
    return msg
