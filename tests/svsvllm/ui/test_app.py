import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest


def test_ui_simple(apptest: AppTest) -> None:
    """Test `ui`."""
    with patch.object(st, "chat_input", return_value="Hi!") as chat_input:
        apptest.run()
        chat_input.assert_called()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
