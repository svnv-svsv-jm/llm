import pytest
import sys
import os
import typing as ty
from loguru import logger

from svsvllm.utils import pop_params_not_in_fn, get_fn_default_value


def _input_fn(arg1: int, arg2: int) -> None:
    """Input callable for testing."""


def _input_fn_def(arg1: int = 1, arg2: int = 2) -> None:
    """Input callable for testing."""


def test_pop_params_not_in_fn() -> None:
    """Test `pop_params_not_in_fn()`."""
    params = {"arg1": 0, "arg2": 0, "will_be_popped": 0}
    params = pop_params_not_in_fn(_input_fn, params)
    logger.info(params)
    assert "arg1" in params
    assert "arg2" in params
    assert "will_be_popped" not in params


def test_get_fn_default_value() -> None:
    """Test `get_fn_default_value`."""
    assert 1 == get_fn_default_value(_input_fn_def, "arg1")
    assert 2 == get_fn_default_value(_input_fn_def, "arg2")
    with pytest.raises(ValueError):
        get_fn_default_value(_input_fn_def, "blabla")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
