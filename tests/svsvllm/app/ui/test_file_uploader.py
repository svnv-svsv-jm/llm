import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest
from io import BytesIO

from svsvllm.settings import settings


@patch.object(settings, "has_chat", False)
def test_file_uploader(
    apptest_ss: AppTest,
    mock_text_file: BytesIO,
) -> None:
    """Test sidebar's file uploader."""
    apptest = apptest_ss
    # Manually set the file in session state (simulate file upload)
    apptest.session_state["uploaded_files"] = [mock_text_file]

    # Run app
    apptest.run()

    # Test callback exists
    assert apptest.session_state["uploaded_files"] is not None


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
