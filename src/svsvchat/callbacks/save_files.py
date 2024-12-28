__all__ = ["SaveFilesCallback"]

import os
import typing as ty
from loguru import logger
from pathlib import Path
import streamlit as st

from svsvchat.settings import settings
from svsvchat.session_state import session_state
from svsvchat.rag import initialize_rag
from ._base import BaseCallback


class SaveFilesCallback(BaseCallback):
    """Save uploaded files to local filesystem."""

    def run(self) -> None:
        """Save uploaded files to local filesystem."""
        # Inform user via warning box
        msg = "Uploading files... Please wait for the success message."
        st.info(msg)
        logger.debug(msg)

        # Get files from session
        logger.trace(f"Getting files from session: {st.session_state}")
        session_state.manual_sync("uploaded_files", reverse=True)
        uploaded_files = session_state.uploaded_files
        logger.trace(f"Got: {uploaded_files}")

        # If no files, return
        if not uploaded_files:
            msg = "No files uploaded."
            logger.trace(msg)
            st.info(msg)
            return

        # Save each file
        msg = "Saving files... Please wait for the success message."
        st.info(msg)
        logger.debug(msg)
        Path(settings.uploaded_files_dir).mkdir(parents=True, exist_ok=True)
        logger.trace(f"Upload location: {settings.uploaded_files_dir}")
        for file in uploaded_files:  # pylint: disable=not-an-iterable
            logger.trace(f"Reading: {file.name}")
            bytes_data = file.read()  # read the content of the file in binary
            filename = os.path.abspath(os.path.join(settings.uploaded_files_dir, file.name))
            Path(filename).touch()
            with open(filename, "wb") as f:
                logger.trace(f"Writing: {filename}")
                f.write(bytes_data)  # write this content elsewhere
                session_state.saved_filenames.append(filename)  # pylint: disable=no-member
        logger.trace(f"Saved files: {session_state.saved_filenames}")
        logger.trace(f"Uploaded files: {session_state.uploaded_files}")
        # Remember saved files
        session_state.manual_sync("saved_filenames", reverse=False)

        # Log success
        msg = "Files uploaded."
        st.info(msg)
        logger.debug(msg)

        # TODO: just vectorize and add new docs to database
        # Re-create RAG?
        if settings.has_chat:
            initialize_rag(force_recreate=True)
