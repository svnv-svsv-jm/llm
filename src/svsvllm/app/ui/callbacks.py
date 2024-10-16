__all__ = ["PageSelectorCallback", "UpdateLanguageCallback"]

import os
import typing as ty
from loguru import logger
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from .rag import initialize_rag
from .const import PageNames, UPLOADED_FILES_DIR
from .utils import get_and_maybe_init_session_state


class BaseCallback:
    """Base callback."""

    def __init__(self, name: str) -> None:
        self.name = name
        # Add instance to state
        if st.session_state.get("callbacks", None) is None:
            logger.trace("Initializing `callbacks` in session state")
            st.session_state["callbacks"] = {}
        logger.trace(f"Adding {self} to session state")
        st.session_state["callbacks"][name] = self


class SaveFilesCallback(BaseCallback):
    """Save uploaded files to local filesystem."""

    def __call__(self, *args: ty.Any, **kwds: ty.Any) -> ty.Any:
        """Save uploaded files to local filesystem."""
        # Inform user via warning box
        msg = "Uploading files... Please wait for the success message."
        st.info(msg)
        logger.debug(msg)

        # Get files from session
        logger.trace("Getting files from session")
        uploaded_files: list[UploadedFile] | None = st.session_state.get("uploaded_files", None)
        logger.trace(f"Got: {uploaded_files}")

        # If no files, return
        if uploaded_files is None:
            logger.trace("No uploaded files.")
            return

        # File manager
        saved_filenames: list[str] = get_and_maybe_init_session_state("saved_filenames", [])
        assert isinstance(saved_filenames, list)

        # Save each file
        msg = "Saving files... Please wait for the success message."
        st.info(msg)
        logger.debug(msg)
        for file in uploaded_files:
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
        msg = "Files uploaded."
        st.info(msg)
        logger.debug(msg)


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
