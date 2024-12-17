__all__ = ["session_state"]

import pytest
from unittest.mock import patch, MagicMock
import sys, os
import typing as ty
from loguru import logger

import streamlit as st

from svsvchat.session_state import SessionState, session_state as ss


def _clear() -> None:
    st.cache_resource.clear()
    st.session_state.clear()
    ss.clear()


@pytest.fixture
def session_state() -> ty.Iterator[SessionState]:
    """Session state."""
    _clear()
    with patch.object(
        ss,
        "reverse",
        True,
    ), patch.object(
        ss,
        "auto_sync",
        False,
    ), patch.object(
        ss,
        "state",
        st.session_state,
    ):
        ss.initialize()
        yield ss
    _clear()
