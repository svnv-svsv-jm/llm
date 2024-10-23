import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

from svsvllm.app.const import ENV_PREFIX
from svsvllm.app.settings import Settings


@pytest.mark.parametrize(
    "key, val",
    [
        ("TEST_MODE", False),
        ("TEST_MODE", True),
        ("uploaded_files_dir", "blabla"),
        ("q_system_prompt", "yo"),
    ],
)
def test_settings(key: str, val: bool) -> None:
    """Test fields."""
    # Temporarily patching os.environ
    with patch.dict(os.environ, {f"{ENV_PREFIX}{key}": f"{val}"}):
        # Create the settings object
        settings = Settings()
        logger.info(f"Settings:\n{settings}")
        # Check value is correct
        assert getattr(settings, key.lower()) == val


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
