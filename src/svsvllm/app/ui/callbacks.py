__all__ = ["PageSelectorCallback", "UpdateLanguageCallback"]

import os
import typing as ty
from loguru import logger
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from .rag import initialize_rag
from .const import PageNames, UPLOADED_FILES_DIR


class BaseCallback:
    """Base callback."""

    def __init__(self, name: str) -> None:
        self.name = name
        # Add instance to state
        if st.session_state.get("callbacks", None):
            st.session_state["callbacks"] = {}
        st.session_state["callbacks"][name] = self


class SaveFilesCallback(BaseCallback):
    """Save uploaded files to local filesystem."""

    def __call__(self, *args: ty.Any, **kwds: ty.Any) -> ty.Any:
        """Save uploaded files to local filesystem."""
        # Inform user via warning box
        st.info("Uploading files... Please wait for the success message.")
        # Get files from session
        files: list[UploadedFile] | None = st.session_state.get("uploaded_files", None)

        # If no files, return
        if files is None:
            return

        # Create file manager
        saved_filenames = st.session_state.get("saved_filenames", [])
        assert isinstance(saved_filenames, list)

        # Save each file
        st.info("Saving files... Please wait for the success message.")
        for file in files:
            logger.trace(f"Reading: {file.name}")
            bytes_data = file.read()  # read the content of the file in binary
            filename = os.path.join(UPLOADED_FILES_DIR, file.name)
            with open(filename, "wb") as f:
                logger.trace(f"Writing: {filename}")
                f.write(bytes_data)  # write this content elsewhere
                saved_filenames.append(filename)

        # Remember saved files
        st.session_state["saved_filenames"] = saved_filenames

        # Re-create RAG?
        if st.session_state.get("has_chat", True):
            initialize_rag(force_recreate=True)
        st.info("Files uploaded!")


class UpdateLanguageCallback(BaseCallback):
    """Callback to update language."""

    def __call__(self) -> None:
        """Update language."""
        logger.trace(
            f"Updating language: {st.session_state.language}->{st.session_state.new_language}"
        )
        st.session_state.language = st.session_state.new_language


class PageSelectorCallback(BaseCallback):
    """Pass this callback to a button, to help change page."""

    def __init__(self, page: str, **kwargs: ty.Any) -> None:
        """
        Args:
            page (str):
                Page name where to land.
        """
        super().__init__(**kwargs)

        # Go to main if unknown page
        if page.lower() not in PageNames.all():
            page = PageNames.MAIN

        # Attribute
        self.page = page

    def __call__(self) -> None:
        """Switch page"""
        logger.trace(f"Switching to {self.page}")
        st.session_state.page = self.page
