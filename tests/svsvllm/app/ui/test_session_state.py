# pylint: disable=unnecessary-dunder-call
import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
import streamlit as st
from streamlit.testing.v1 import AppTest
from pydantic_core import ValidationError

from svsvllm.utils.singleton import Singleton
from svsvllm.app.ui.session_state import SessionState, _SessionState, _VerboseSetItem


@pytest.mark.parametrize("value", [False, True])
def test_session_state_integration_w_apptest(
    apptest_ss: AppTest,
    session_state: SessionState,
    value: bool,
) -> None:
    """Test session state integration with `AppTest`."""
    apptest_ss.session_state["has_chat"] = value
    assert session_state.state.has_chat == value


def test_reverse_sync() -> None:
    """Test that reverse sync is called when the flag is set."""
    # Reset to be able to create a new one
    SessionState.reset()
    # Start test
    logger.info("Testing reverse sync is called")
    with patch.object(
        _SessionState,
        "_sync_reverse",
    ) as mock:
        SessionState(reverse=True)
        mock.assert_called()


def test_session_state_is_singleton() -> None:
    """Test there is only one instance always."""
    ss0 = SessionState(reverse=True)
    ss1 = SessionState(reverse=False)
    assert ss0 is ss1, f"{Singleton._instances}"


@pytest.mark.parametrize("state_class", [SessionState, _SessionState])
@pytest.mark.parametrize("bind", [False, True])
@pytest.mark.parametrize("check_reverse_sync", [False, True])
@pytest.mark.parametrize("use_reverse_sync", [False, True])
@pytest.mark.parametrize(
    "field, value, validation_fails",
    [
        ("has_chat", 3, True),
        ("has_chat", False, False),
        ("has_chat", True, False),
        ("new_language", "English", False),
        ("openai_model_name", "what", False),
        ("page", "yo", True),
        ("openai_model_name", "what", False),
        ("saved_filenames", ["yoyo"], False),
        ("openai_client", 5, True),
    ],
)
def test_session_states_are_synced(
    apptest: AppTest,
    state_class: ty.Type[SessionState | _SessionState],
    bind: bool,
    field: str,
    value: ty.Any,
    validation_fails: bool,
    check_reverse_sync: bool,
    use_reverse_sync: bool,
) -> None:
    """Test `SessionState` is always synced with `st.session_state`."""
    # Reset to be able to create a new one
    SessionState.reset()

    # Reverse sync?
    params = dict()
    if state_class is SessionState:
        params = dict(reverse=use_reverse_sync)

    # Create state
    our_state = state_class(**params)  # type: ignore
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
            logger.info(f"Updating `{field}` with `{value}`")
            if not check_reverse_sync:
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
        logger.info(f"(check_reverse_sync={check_reverse_sync}) Updating `{field}` with `{value}`")
        with _VerboseSetItem(depth=6):
            if check_reverse_sync:
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
