__all__ = ["app_main_file", "apptest"]

import pytest
from unittest.mock import patch, MagicMock
import sys, os
import typing as ty
from loguru import logger

from streamlit.testing.v1 import AppTest
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from svsvchat.settings import settings
from svsvchat import __main__


@pytest.fixture(autouse=True)
def reset_state() -> None:
    """Reset Streamlit's session state."""
    st.cache_resource.clear()
    st.session_state.clear()
    ctx = get_script_run_ctx()
    if ctx is not None:
        ctx.session_state.clear()


@pytest.fixture
def app_main_file() -> str:
    """App file."""
    path = os.path.abspath(__main__.__file__)
    logger.debug(f"Loading script: {path}")
    return str(path)


@pytest.fixture
def apptest(
    trace_logging_level: bool,
    app_main_file: str,
) -> ty.Iterator[AppTest]:
    """App for testing."""
    at = AppTest.from_file(app_main_file, default_timeout=30)

    # Keep ref to original method
    __run = at.run

    def run(**kwags: ty.Any) -> AppTest | None:
        """Wrap original method in a `try-except` statement."""
        logger.debug("Running patched `run`")
        try:
            return __run(**kwags)
        except RuntimeError as e:
            logger.debug(e)
            return None

    # Patch the run method to time out safely
    at.run = run  # type: ignore

    with patch.object(settings, "test_mode", True):
        yield at

    at.run = __run
