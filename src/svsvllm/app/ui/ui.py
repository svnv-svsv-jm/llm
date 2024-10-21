__all__ = ["ui"]

from loguru import logger
import streamlit as st

from .const import PageNames
from .pages import settings_page, main_page
from .session_state import get_state


# TODO: check https://docs.streamlit.io/develop/concepts/app-testing/get-started
def ui() -> None:
    """App UI."""
    # Display content based on the current page
    page = get_state("page")
    logger.trace(f"Display content based on the current page: {page}")
    if page == PageNames.SETTINGS:
        settings_page()
    else:  # st.session_state.page == PageNames.MAIN
        main_page()
