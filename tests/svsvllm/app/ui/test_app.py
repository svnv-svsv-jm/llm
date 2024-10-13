import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest

# TODO: see https://medium.com/@chrisschneider/build-a-high-quality-streamlit-app-with-test-driven-development-eef4e462f65e


def test_app_starts(apptest: AppTest) -> None:
    """Verify the app starts without errors."""
    apptest.session_state["has_chat"] = False
    apptest.run()
    assert not apptest.exception


def test_page_title(apptest: AppTest) -> None:
    """Verify the app has the expected title."""
    apptest.session_state["has_chat"] = False
    apptest.run()
    assert len(apptest.subheader) == 1
    assert "Smart Assistant" in apptest.subheader[0].value
    assert len(apptest.caption) > 0
    assert len(apptest.title) > 0
    assert "FiscalAI" in apptest.title[0].value


def test_ui_simple(apptest: AppTest) -> None:
    """Test `app`."""
    with patch.object(st, "chat_input", side_effect=["Hello", None]) as chat_input:
        apptest.session_state["openai_api_key"] = None
        apptest.run()
        chat_input.assert_called()
    assert apptest.session_state.openai_api_key is None


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
