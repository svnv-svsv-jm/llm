__all__ = ["ui"]


def ui() -> None:
    """App UI."""
    # Imports here, to make this func self sufficient during tests
    from loguru import logger
    import streamlit as st

    from svsvllm.app.const import PageNames
    from svsvllm.app.ui.pages import settings_page, main_page
    from svsvllm.app.ui.session_state import SessionState

    # Initialize session state
    state = SessionState().state
    # Display content based on the current page
    page = state.page
    logger.debug(f"Display content based on the current page: {page}")
    if page == PageNames.SETTINGS:
        settings_page()
    else:  # st.session_state.page == PageNames.MAIN
        main_page()
    logger.trace("End of UI.")
