import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

from svsvchat.callbacks import UpdateLanguageCallback
from svsvchat.session_state import SessionState


@pytest.mark.parametrize("page", ["main", "settings"])
def test_PageSelectorCallback(
    session_state: SessionState,
    page: str,
) -> None:
    """Test `PageSelectorCallback`."""
    session_state.language = "English"
    session_state.new_language = "Italian"
    cb = UpdateLanguageCallback(name="yo")
    cb()
    logger.info(f"Callback: {cb}")
    assert session_state.language == session_state.new_language


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
