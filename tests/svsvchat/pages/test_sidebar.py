import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
from streamlit.testing.v1 import AppTest

from svsvchat.callbacks import PageSelectorCallback, UpdateLanguageCallback
from svsvchat.session_state import SessionState
from svsvchat.settings import settings


def _has_cb(
    session_state: SessionState,
    cb_class: type[PageSelectorCallback | UpdateLanguageCallback],
) -> None:
    has_cb = False
    for _, cb in session_state.callbacks.items():
        if isinstance(cb, cb_class):
            has_cb = True
            cb.run()
            break
    assert has_cb


def test_sidebar(apptest: AppTest, session_state: SessionState) -> None:
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
    _has_cb(session_state, PageSelectorCallback)
    _has_cb(session_state, UpdateLanguageCallback)

    # Success
    logger.success("Ok.")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
