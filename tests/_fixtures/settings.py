__all__ = ["settings"]

import pytest
from unittest.mock import patch, MagicMock
import sys, os
import typing as ty
from loguru import logger

import streamlit as st

from svsvchat.settings import Settings, settings as sttng


@pytest.fixture
def settings() -> ty.Iterator[Settings]:
    """App settings."""
    yield sttng
