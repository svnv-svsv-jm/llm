import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest

from svsvchat.session_state import SessionState


def test_settings_page(session_state: SessionState, apptest: AppTest) -> None:
    """Verify we can choose the settings page."""
    session_state.page = "settings"
    apptest.run()

    # Test basics
    assert not apptest.exception

    # Test number_input
    logger.info(apptest.number_input)
    assert len(apptest.number_input) > 0

    # Test buttons
    logger.info(apptest.button)
    assert len(apptest.button) > 0


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
