import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
from streamlit.runtime.uploaded_file_manager import UploadedFile

from svsvchat.callbacks import SaveFilesCallback
from svsvchat.session_state import SessionState
from svsvchat.settings import Settings


@pytest.mark.parametrize("no_files", [True, False])
def test_SaveFilesCallback(
    session_state: SessionState,
    settings: Settings,
    res_docs_path: str,
    uploaded_files: list[UploadedFile],
    no_files: bool,
) -> None:
    """Test `SaveFilesCallback`."""
    # Call callback
    _uploaded_files = [] if no_files else uploaded_files
    with patch.object(
        session_state,
        "uploaded_files",
        _uploaded_files,
    ), patch.object(
        settings,
        "uploaded_files_dir",
        res_docs_path,
    ):
        cb = SaveFilesCallback(name="cb")
        cb()
        logger.info(f"Callback: {cb}")

    # Test no saved files if `no_files == True`
    if no_files:
        assert session_state.saved_filenames == []
        return

    # Test uploaded files are there
    assert len(session_state.saved_filenames) == len(uploaded_files)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
