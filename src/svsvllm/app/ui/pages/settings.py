__all__ = ["settings_page"]

from loguru import logger
import streamlit as st

from ..sidebar import sidebar
from ..start_messages import START_MSG_EN
from ..const import PageNames
from ..callbacks import PageSelectorCallback


def settings_page() -> None:
    """Function to display the settings page content."""
    st.title("Settings")
    st.write("Here you can modify the settings.")

    # # Add some settings widgets (for example, a language selector)
    # language = st.selectbox("Select Language", ["English", "Spanish", "French"])
    # st.write(f"Selected Language: {language}")

    # Button to go back to the main page
    st.button("Go to Main Page", on_click=PageSelectorCallback(PageNames.MAIN))
