__all__ = ["ui"]

import streamlit as st

from .response import get_response
from .sidebar import sidebar
from .start_messages import START_MSG_EN


# TODO: check https://docs.streamlit.io/develop/concepts/app-testing/get-started
def ui() -> None:
    """App UI."""

    # Sidebar
    sidebar_container = sidebar()

    # Title and initialization
    st.title("💬 FiscalAI")
    st.caption("🚀 Your favorite chatbot, powered by FiscalAI.")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": START_MSG_EN,
            },
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Main chat loop
    if prompt := st.chat_input():
        # Start session
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # LLM's response
        msg = get_response(sidebar_container["openai_api_key"])

        # Update session
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
