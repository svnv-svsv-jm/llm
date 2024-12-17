__all__ = ["session_state"]

import pytest
import sys, os
import typing as ty
from loguru import logger

from svsvchat.session_state import SessionState, session_state as ss


@pytest.fixture
def session_state() -> SessionState:
    """App's session state."""
    return ss
