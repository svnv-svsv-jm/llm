import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest


def test_dummy_app(dummy_at: AppTest) -> None:
    """This test exists to keep a minimal example of a clean test:

    * The `AppTest` object does not get stuck on `run()`.
    * All required/expected `ElementList` objects are not empty.
    """
    dummy_at.run()

    # Test basics
    assert not dummy_at.exception

    # Test buttons
    logger.info(dummy_at.title)
    assert len(dummy_at.title) > 0


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
