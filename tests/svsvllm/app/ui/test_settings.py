import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest

from svsvllm.app.ui.const import PageNames


def test_settings_page(apptest: AppTest) -> None:
    """Verify we can choose the settings page."""
    apptest.session_state["page"] = PageNames.SETTINGS
    apptest.run()
    assert not apptest.exception
    assert apptest.session_state.page == PageNames.SETTINGS


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
