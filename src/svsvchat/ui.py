__all__ = ["ui"]

from loguru import logger
import streamlit as st

from .pages import settings_page, chat_page


def ui() -> None:
    """App UI."""
    # Display content based on the current page
    page = st.session_state["page"]
    logger.debug(f"Display content based on the current page: {page}")
    if page == "settings":
        settings_page()
    else:  # st.session_state.page == PageNames.MAIN
        chat_page()
    logger.trace("End of UI.")
