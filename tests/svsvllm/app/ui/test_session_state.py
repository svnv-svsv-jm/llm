# pylint: disable=unnecessary-dunder-call
import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest
from pydantic_core import ValidationError

from svsvllm.app.ui.session_state import SessionState


@pytest.mark.parametrize("bind", [False, True])
@pytest.mark.parametrize(
    "field, value, validation_fails, reverse_sync",
    [
        ("has_chat", False, False, False),
        ("has_chat", True, False, False),
        ("has_chat", 3, True, False),
        ("new_language", "English", False, False),
        ("openai_model_name", "what", False, False),
        ("page", "yo", True, False),
        ("openai_model_name", "what", False, True),
    ],
)
def test_session_states_are_synced(
    apptest: AppTest,
    bind: bool,
    field: str,
    value: ty.Any,
    validation_fails: bool,
    reverse_sync: bool,
) -> None:
    """Test `SessionState` is always synced with `st.session_state`."""
    # Create state
    our_state = SessionState()
    logger.info(f"Creating session state: {type(our_state)}")

    # Bind
    if bind:
        our_state.bind(apptest.session_state)
        st_state = apptest.session_state
    else:
        our_state.unbind()
        st_state = st.session_state  # type: ignore

    # Here we are passing an invalid value and we expect a ValidationError
    if validation_fails:
        with pytest.raises(ValidationError):
            setattr(our_state, field, value)
        return

    # Update field
    if reverse_sync:
        with patch.object(our_state, "__setattr__", side_effect=our_state.__setattr__) as mock:
            st_state[field] = value
            # Test our class method was called even if we set the state directly
            mock.assert_called()
    else:
        setattr(our_state, field, value)

    # Test they stay synced
    assert getattr(st_state, field) == value
    assert getattr(our_state, field) == value
    assert our_state[field] == st_state[field]


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    logger.add("pytest_artifacts/test.log", level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
