__all__ = ["ui"]

from loguru import logger
import streamlit as st

from .const import PageNames
from .pages import settings_page, main_page


# TODO: check https://docs.streamlit.io/develop/concepts/app-testing/get-started
def ui() -> None:
    """App UI."""
    # Initialize session state for page navigation
    if "page" not in st.session_state:
        logger.trace("Initializing session state for page")
        st.session_state.page = PageNames.MAIN  # Default page is the main page

    # Display content based on the current page
    page = st.session_state.page
    logger.trace(f"Display content based on the current page: {page}")
    if page == PageNames.SETTINGS:
        settings_page()
    else:  # st.session_state.page == PageNames.MAIN
        main_page()
