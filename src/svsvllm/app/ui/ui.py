__all__ = ["ui"]

from loguru import logger
import streamlit as st

from .const import PageNames
from .pages import settings_page, main_page
from .session_state import SessionState


def ui() -> None:
    """App UI."""
    # Initialize session state
    state = SessionState().state
    # Display content based on the current page
    page = state.page
    logger.debug(f"Display content based on the current page: {page}")
    if page == PageNames.SETTINGS:
        settings_page()
    else:  # st.session_state.page == PageNames.MAIN
        main_page()
