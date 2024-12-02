__all__ = ["ui"]


def ui() -> None:
    """App UI."""
    # Imports here, to make this func self sufficient during tests
    from loguru import logger
    import streamlit as st

    from svsvllm.const import PageNames
    from svsvllm.ui.pages import settings_page, main_page
    from svsvllm.ui.session_state import SessionState

    # Initialize session state
    # NOTE: Run this regardless. The `SessionState` needs to be initialized asap so that it is bound to Streamlit's session state
    state = SessionState().state

    # Display content based on the current page
    page = state.page
    logger.debug(f"Display content based on the current page: {page}")
    if page == PageNames.SETTINGS:
        settings_page()
    else:  # st.session_state.page == PageNames.MAIN
        main_page()
    logger.trace("End of UI.")
