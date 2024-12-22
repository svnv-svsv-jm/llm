__all__ = ["file_uploader"]

import os
from loguru import logger
import streamlit as st

from svsvchat.settings import settings
from svsvchat.callbacks import SaveFilesCallback


def file_uploader(key: str = None) -> None:
    """Adds the widget that lets user upload files and save them to disk."""
    if key is None:
        key = settings.uploaded_files_key
    logger.trace("Creating file uploader")
    if key in st.session_state:
        logger.trace(f"Clearing key: '{key}'")
        del st.session_state[key]
    uploaded_files = st.file_uploader(
        settings.file_uploader_label,
        type=settings.allowed_rag_file_extensions,
        accept_multiple_files=True,
        key=key,
        on_change=SaveFilesCallback("save-file"),
        help="Uploade documents that this chatbot may want to read.",
    )
    if uploaded_files:
        st.success(f"Uploaded file: {uploaded_files}")
