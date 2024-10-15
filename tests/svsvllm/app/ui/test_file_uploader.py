import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
from streamlit.testing.v1 import AppTest
from io import BytesIO


def test_file_uploader(apptest: AppTest, mock_text_file: BytesIO) -> None:
    """Test sidebar's file uplodaer works."""
    # No need for chat
    apptest.session_state["has_chat"] = False

    # Manually set the file in session state (simulate file upload)
    apptest.session_state["uploaded_files"] = [mock_text_file]

    # Run app
    apptest.run()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
