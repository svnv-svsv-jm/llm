__all__ = ["file_uploader"]

import os
from loguru import logger
import streamlit as st

from .callbacks import SaveFilesCallback


def file_uploader() -> None:
    """Let user upload files and save them to disk."""
    logger.trace("Creating file uploader")
    st.file_uploader(
        "Upload files from a folder",
        accept_multiple_files=True,
        key="uploaded_files",
        on_change=SaveFilesCallback("save-file"),
    )
