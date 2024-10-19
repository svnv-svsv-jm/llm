# pylint: disable=unnecessary-dunder-call
import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest
from pydantic_core import ValidationError

from svsvllm.app.ui.session_state import SessionState, _SessionState


def test_session_state_is_singleton() -> None:
    """Test there is only one instance always."""
    assert SessionState() is SessionState()


@pytest.mark.parametrize("state_class", [SessionState, _SessionState])
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
        ("saved_filenames", ["yoyo"], False, True),
        ("openai_client", 5, True, True),
    ],
)
def test_session_states_are_synced(
    apptest: AppTest,
    state_class: ty.Type[SessionState | _SessionState],
    bind: bool,
    field: str,
    value: ty.Any,
    validation_fails: bool,
    reverse_sync: bool,
) -> None:
    """Test `SessionState` is always synced with `st.session_state`."""
    # Create state
    our_state = state_class()
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
            if not reverse_sync:
                setattr(our_state, field, value)
            else:
                st_state[field] = value
        return

    # Patch with itself, just to assert it is called
    obj = our_state._state if isinstance(our_state, SessionState) else our_state
    assert isinstance(obj, _SessionState)
    with patch.object(
        obj,
        "__setattr__",
        side_effect=obj.__setattr__,
    ) as mock:

        # Update field
        if reverse_sync:
            st_state[field] = value
        else:
            setattr(our_state, field, value)

        # Test our class method was called even if we set the st.state directly, not via `our_state`
        mock.assert_called()

    # Test they stay synced
    assert getattr(st_state, field) == value
    assert getattr(our_state, field) == value
    assert our_state[field] == st_state[field]


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    logger.add("pytest_artifacts/test.log", level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
