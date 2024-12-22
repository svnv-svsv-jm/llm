import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
from streamlit.testing.v1 import AppTest
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pydantic_core._pydantic_core import ValidationError  # pylint: disable=no-name-in-module

from svsvchat.callbacks import SaveFilesCallback
from svsvchat.settings import settings
from svsvchat.panels.file_upload import file_uploader


class FileUploader:
    """Patcher, to force returning something."""

    def __init__(self, files: list[UploadedFile]) -> None:
        self.files = files
        self._file_uploader = st._main.file_uploader

    def __call__(self, *args: ty.Any, **kwargs: ty.Any) -> list[UploadedFile]:
        self._file_uploader(*args, **kwargs)
        return self.files


def test_file_uploader(check_state_has_cb: ty.Callable, uploaded_files: UploadedFile) -> None:
    """Test function directly."""
    st.session_state["uploaded_files"] = uploaded_files
    with patch.object(st, "file_uploader", side_effect=FileUploader(uploaded_files)):
        file_uploader()
    # Test callback is there
    check_state_has_cb(SaveFilesCallback)
    # Success
    logger.success("Ok.")


def test_file_uploader_app(
    apptest: AppTest,
    check_state_has_cb: ty.Callable,
    uploaded_files: UploadedFile,
) -> None:
    """Test we can uploader a file and that this file is then present in the `session_state`."""
    # Populate `uploaded_files` key to test it is effectively cleared by the app
    # Also quickly check that not setting this key to a `list[UploadedFile]` object raises a `ValidationError`
    with pytest.raises(ValidationError):
        st.session_state["uploaded_files"] = None
    st.session_state["uploaded_files"] = uploaded_files

    # Run
    with patch.object(settings, "has_chat", False):
        apptest.run(timeout=5)
    logger.info(f"App: {apptest}")

    # Test basics
    assert not apptest.exception

    # Test callback is there
    check_state_has_cb(SaveFilesCallback)

    # Success
    logger.success("Ok.")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
