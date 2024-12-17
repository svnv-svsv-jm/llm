__all__ = ["app_main_file", "apptest"]

import pytest
from unittest.mock import patch, MagicMock
import sys, os
import typing as ty
from loguru import logger

import streamlit as st

from svsvchat.settings import settings


@pytest.fixture
def app_main_file() -> str:
    """App file."""
    path = os.path.abspath(main.__file__)
    logger.debug(f"Loading script: {path}")
    return path


@pytest.fixture
def apptest(trace_logging_level: bool, app_main_file: str) -> ty.Iterator[AppTest]:
    """App for testing."""
    with patch.object(settings, "test_mode", True):
        at = AppTest.from_file(app_main_file, default_timeout=30)
        yield at
