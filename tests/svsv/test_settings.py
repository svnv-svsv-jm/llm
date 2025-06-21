import os
from unittest.mock import patch

import pydantic_settings
import pytest
from loguru import logger

import svsv


def test_settings(settings: svsv.Settings) -> None:
    """Test `settings` exists."""
    logger.info(settings)
    assert isinstance(settings, pydantic_settings.BaseSettings)


def test_settings_from_env_vars() -> None:
    """Test we can set settings from environment variables."""
    with patch.dict(os.environ, {"SVSV_DEBUG_MODE": "True"}):
        s = svsv.Settings()
        assert s.debug_mode


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
