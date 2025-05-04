__all__ = ["run_ui"]

import streamlit as st


def run_ui() -> None:
    """Run UI."""
    st.title("Chat App")
    user_input = st.chat_input("Type a message")
    if user_input:
        st.write(f"You said: {user_input}")


if __name__ == "__main__":
    run_ui()
