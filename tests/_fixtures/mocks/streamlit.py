__all__ = ["mock_chat_input"]

import pytest
from unittest.mock import MagicMock, patch
import os
import typing as ty
from loguru import logger

import streamlit as st


@pytest.fixture
def mock_chat_input() -> ty.Iterator[MagicMock]:
    """Mock first few user inputs."""
    with patch.object(st, "chat_input", side_effect=["Hello", None]) as chat_input:
        yield chat_input
