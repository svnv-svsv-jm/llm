import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

from svsvchat.callbacks import PageSelectorCallback
from svsvchat.session_state import SessionState


@pytest.mark.parametrize("page", ["main", "settings"])
def test_PageSelectorCallback(
    session_state: SessionState,
    page: str,
) -> None:
    """Test `PageSelectorCallback`."""
    cb = PageSelectorCallback(page, name="page-selector")
    cb()
    logger.info(f"Callback: {cb}")
    assert session_state.page == page


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
