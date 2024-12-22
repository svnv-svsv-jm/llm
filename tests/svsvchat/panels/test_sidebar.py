import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
from streamlit.testing.v1 import AppTest

from svsvchat.callbacks import PageSelectorCallback, UpdateLanguageCallback, SaveFilesCallback
from svsvchat.session_state import SessionState
from svsvchat.settings import settings


def test_sidebar(
    apptest: AppTest,
    session_state: SessionState,
    check_state_has_cb: ty.Callable,
) -> None:
    """Test `main` page setup: title, headers, etc."""
    # Run
    with patch.object(settings, "has_chat", False):
        apptest.run(timeout=5)
    logger.info(f"App: {apptest}")

    # Test basics
    assert not apptest.exception

    # Test
    logger.info(f"Sidebar: {apptest.sidebar}")
    assert apptest.sidebar.selectbox
    assert apptest.sidebar.selectbox.values
    logger.info(f"Button: {apptest.sidebar.button}")
    assert apptest.sidebar.button

    # Test callbacks
    logger.info(session_state.callbacks)
    check_state_has_cb(PageSelectorCallback)
    check_state_has_cb(UpdateLanguageCallback)
    check_state_has_cb(SaveFilesCallback)

    # Success
    logger.success("Ok.")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
