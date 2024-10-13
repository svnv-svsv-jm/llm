import pytest
import os
from loguru import logger
from streamlit.testing.v1 import AppTest

import svsvllm.__main__ as main


@pytest.fixture
def apptest() -> AppTest:
    """App for testing."""
    path = os.path.abspath(main.__file__)
    logger.debug(f"Loading script: {path}")
    at = AppTest.from_file(path, default_timeout=30)
    return at
