import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest
from io import BytesIO

from svsvllm.app.ui.callbacks import SaveFilesCallback


def test_savefiles_callback(apptest: AppTest, mock_text_file: BytesIO) -> None:
    """Test callback works."""
    # No need for chat
    apptest.session_state["has_chat"] = False

    # Manually set the file in session state (simulate file upload)
    apptest.session_state["uploaded_files"] = [mock_text_file]

    # Run app
    apptest.run()

    # Test callback exists
    assert apptest.session_state["uploaded_files"] is not None
    callback = apptest.session_state["callbacks"]["save-file"]
    assert isinstance(callback, SaveFilesCallback)

    # Test: we are calling the callback manually here, so it will interact with `st.session_state`, not with `apptest.session_state`
    st.session_state["has_chat"] = False
    st.session_state["uploaded_files"] = [mock_text_file]
    uploaded_files = st.session_state["uploaded_files"]
    assert uploaded_files is not None
    callback()
    saved_filenames = st.session_state["saved_filenames"]
    assert saved_filenames is not None
    for file, name in zip(uploaded_files, saved_filenames):
        assert file.name in name


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
