import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
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

    # Test
    callback = apptest.session_state["callbacks"]["save-file"]
    assert isinstance(callback, SaveFilesCallback)
    callback()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
