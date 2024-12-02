__all__ = ["PageSelectorCallback", "UpdateLanguageCallback", "VectorizeCallback"]

import os
import typing as ty
from loguru import logger
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from io import BytesIO

from svsvllm.settings import settings
from svsvllm.const import PageNames
from .rag import initialize_rag, create_history_aware_retriever
from .session_state import SessionState


class BaseCallback:
    """Base callback."""

    def __init__(self, name: str) -> None:
        self._name = name
        state = SessionState()
        st.session_state.setdefault("callbacks", state["callbacks"])
        logger.trace(f"Adding {self.name} to session state.")
        state["callbacks"][name] = self
        logger.trace(f"Added {self.name} to session state: {st.session_state}")

    @property
    def name(self) -> str:
        """Name of the callback."""
        nature = self.__class__.__name__
        name = self._name
        return f"{nature}({name})"

    def __repr__(self) -> str:
        return self.name

    def __call__(self) -> None:
        """Main caller."""
        logger.trace(f"Running {self.name}")
        self.run()

    def run(self) -> None:
        """Subclass method."""
        raise NotImplementedError()


class VectorizeCallback(BaseCallback):
    """Re-create the RAG with the new settings."""

    def run(self) -> None:
        """Re-create the RAG with the new settings."""
        create_history_aware_retriever(force_recreate=True)


class SaveFilesCallback(BaseCallback):
    """Save uploaded files to local filesystem."""

    def run(self) -> None:
        """Save uploaded files to local filesystem."""
        # Inform user via warning box
        msg = "Uploading files... Please wait for the success message."
        st.info(msg)
        logger.debug(msg)

        # Get files from session
        logger.trace("Getting files from session")
        uploaded_files = SessionState().state.uploaded_files
        logger.trace(f"Got: {uploaded_files}")

        # If no files, return
        if not uploaded_files:
            logger.trace("No uploaded files.")
            return

        # File manager
        saved_filenames: list[str] = SessionState().state.saved_filenames
        assert isinstance(saved_filenames, list)

        # Save each file
        msg = "Saving files... Please wait for the success message."
        st.info(msg)
        logger.debug(msg)
        for file in uploaded_files:
            logger.trace(f"Reading: {file.name}")
            bytes_data = file.read()  # read the content of the file in binary
            filename = os.path.join(settings.uploaded_files_dir, file.name)
            with open(filename, "wb") as f:
                logger.trace(f"Writing: {filename}")
                f.write(bytes_data)  # write this content elsewhere
                saved_filenames.append(filename)

        # Remember saved files
        st.session_state["saved_filenames"] = saved_filenames

        # TODO: just vectorize and add new docs to database
        # Re-create RAG?
        if settings.has_chat:
            initialize_rag(force_recreate=True)
        msg = "Files uploaded."
        st.info(msg)
        logger.debug(msg)


class UpdateLanguageCallback(BaseCallback):
    """Callback to update language."""

    def run(self) -> None:
        """Update language."""
        logger.trace(
            f"Updating language: {SessionState().state.language}->{SessionState().state.new_language}"
        )
        st.session_state["language"] = SessionState().state.new_language


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

    def run(self) -> None:
        """Switch page"""
        logger.trace(f"Switching to {self.page}")
        st.session_state["page"] = self.page
