__all__ = ["settings_page"]

from loguru import logger
import streamlit as st

from svsvchat.callbacks import PageSelectorCallback
from svsvchat.session_state import session_state


def settings_page() -> None:
    """Function to display the settings page content."""
    logger.trace(f"Setting settings page.")
    st.title("Settings")
    st.write("Here you can modify the settings.")

    # Button to go back to the main page
    st.button(
        "Go to Main Page",
        on_click=PageSelectorCallback("main", name="page-selector"),
    )

    # Settings for the database
    st.number_input(
        "Chunk size.",
        min_value=1,
        step=1,
        value=session_state.chunk_size,
        format="%1f",
        key="chunk_size",
        help="Input a integer number to select the chunk size for the RAG.",
        on_change=None,
    )
    st.number_input(
        "Chunk overlap.",
        min_value=1,
        step=1,
        value=session_state.chunk_overlap,
        format="%1f",
        key="chunk_overlap",
        help="Input a integer number to select the chunk overlap for the RAG.",
        on_change=None,
    )

    # # Button to revectorize the databse with the new settings
    # st.button(
    #     "Vectorize the databse with the new settings",
    #     on_click=VectorizeCallback(name="vectorize"),
    # )
