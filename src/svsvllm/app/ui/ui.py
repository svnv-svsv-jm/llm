__all__ = ["ui"]

import streamlit as st

from .response import get_response
from .sidebar import sidebar
from .start_messages import START_MSG_EN
from .const import PageNames
from .callbacks import PageSelectorCallback


def main_page() -> None:
    """Main page."""
    # Sidebar
    sidebar()

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
        msg = get_response(
            model=st.session_state.model,
            openai_api_key=st.session_state.openai_api_key,
        )

        # Update session
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)


def settings_page() -> None:
    """Function to display the settings page content."""
    st.title("Settings")
    st.write("Here you can modify the settings.")

    # # Add some settings widgets (for example, a language selector)
    # language = st.selectbox("Select Language", ["English", "Spanish", "French"])
    # st.write(f"Selected Language: {language}")

    # Button to go back to the main page
    st.button("Go to Main Page", on_click=PageSelectorCallback(PageNames.MAIN))


# TODO: check https://docs.streamlit.io/develop/concepts/app-testing/get-started
def ui() -> None:
    """App UI."""
    # Initialize session state for page navigation
    if "page" not in st.session_state:
        st.session_state.page = PageNames.MAIN  # Default page is the main page

    # Display content based on the current page
    if st.session_state.page == PageNames.MAIN:
        main_page()
    elif st.session_state.page == PageNames.SETTINGS:
        settings_page()
    else:
        main_page()
