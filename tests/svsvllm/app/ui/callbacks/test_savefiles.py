import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest
from io import BytesIO

from svsvllm.settings import settings
from svsvllm.ui.callbacks import SaveFilesCallback


@patch.object(settings, "verbose_item_set", True)
@patch.object(settings, "has_chat", False)
def test_savefiles_callback(apptest_ss: AppTest, mock_text_file: BytesIO) -> None:
    """Test callback works."""
    apptest = apptest_ss
    # Run app
    apptest.run()
    for ex in apptest.exception:
        logger.error(ex)
    n_errors = len(apptest.exception)
    assert n_errors == 0

    # Test callback exists
    assert apptest.session_state["uploaded_files"] is not None
    callback = apptest.session_state["callbacks"]["save-file"]
    assert isinstance(callback, SaveFilesCallback)

    # Test: we are calling the callback manually here, so it will interact with `st.session_state`, not with `apptest.session_state`
    uploaded_files = [mock_text_file]
    st.session_state["uploaded_files"] = uploaded_files
    callback()
    saved_filenames = st.session_state["saved_filenames"]
    assert saved_filenames is not None
    for file, name in zip(uploaded_files, saved_filenames):
        assert file.name in name


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
