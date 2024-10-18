import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest
from pydantic_core import ValidationError

from svsvllm.app.ui.session_state import SessionState


# def test_session_state_is_singleton(apptest: AppTest) -> None:
#     """Test there is only one instance."""
#     assert SessionState() is SessionState()
#     assert SessionState().session_state is SessionState().session_state
#     state = SessionState()
#     state.bind(apptest.session_state)
#     assert SessionState().session_state is SessionState().session_state


@pytest.mark.parametrize("bind", [False, True])
@pytest.mark.parametrize(
    "field, value, validation_fails",
    [
        ("has_chat", False, False),
        ("has_chat", True, False),
        ("has_chat", 3, True),
        ("new_language", "English", False),
        ("openai_model_name", "what", False),
        ("page", "yo", True),
    ],
)
def test_session_states_are_synced(
    apptest: AppTest,
    bind: bool,
    field: str,
    value: ty.Any,
    validation_fails: bool,
) -> None:
    """Test `SessionState` is always synced with `st.session_state`."""
    # Create state
    state = SessionState()
    logger.info(f"Creating session state: {type(state)}")

    # Bind
    if bind:
        state.bind(apptest.session_state)
        ss = apptest.session_state
    else:
        state.unbind()
        ss = st.session_state  # type: ignore

    # Here we are passing an invalid value and we expect a ValidationError
    if validation_fails:
        with pytest.raises(ValidationError):
            setattr(state, field, value)
        return

    # Mess around with 'field'
    setattr(state, field, value)
    assert ss[field] == getattr(state, field)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    logger.add("pytest_artifacts/test.log", level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
