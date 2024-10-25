__all__ = ["settings_page"]

from loguru import logger
import streamlit as st

from svsvllm.app.const import PageNames
from ..callbacks import PageSelectorCallback, VectorizeCallback


def settings_page() -> None:
    """Function to display the settings page content."""
    logger.trace(f"Setting settings page.")
    st.title("Settings")
    st.write("Here you can modify the settings.")

    # Button to go back to the main page
    st.button(
        "Go to Main Page",
        on_click=PageSelectorCallback(
            PageNames.MAIN,
            name="page-selector",
        ),
    )

    # Settings for the database
    st.number_input(
        "Chunk size.",
        min_value=1,
        value="min",
        format="%1f",
        key="chunk_size",
        help="Input a integer number.",
        on_change=None,
    )
    st.number_input(
        "Chunk overlap.",
        min_value=1,
        value="min",
        format="%1f",
        key="chunk_overlap",
        help="Input a integer number.",
        on_change=None,
    )
    # Button to revectorize the databse with the new settings
    st.button(
        "Vectorize the databse with the new settings",
        on_click=VectorizeCallback(name="vectorize"),
    )
