__all__ = ["run_ui"]

import streamlit as st


def run_ui() -> None:
    """Run UI."""
    # Title
    st.title("Chat App")

    # Accept user input
    if prompt := st.chat_input("What's up?"):

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)


if __name__ == "__main__":
    run_ui()
