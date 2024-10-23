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


@pytest.mark.parametrize("auto_sync", [False, True])
@pytest.mark.parametrize("key", ["language", "openai_api_key", "callbacks"])
def test_accessing_session_state(key: str, auto_sync: bool) -> None:
    """Test we can access the session state at any key without errors because they are initialized by default."""
    logger.info("Starting...")
    with SessionState(auto_sync=auto_sync) as session_state:
        value = session_state[key]
        logger.info(f"{key}: {value}")


@pytest.mark.parametrize("value", ["Italian", "English"])
def test_session_state_integration_w_apptest(
    apptest_ss: AppTest,
    session_state: SessionState,
    value: bool,
) -> None:
    """Test session state integration with `AppTest`."""
    apptest_ss.session_state["language"] = value
    assert session_state.state.language == value


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
        with SessionState(reverse=True, auto_sync=True):
            mock.assert_called()


def test_session_state_is_singleton(session_state: SessionState) -> None:
    """Test there is only one instance always."""
    with SessionState(reverse=False) as ss:
        assert session_state is ss, f"{Singleton._instances}"


@pytest.mark.parametrize("auto_sync", [False, True])
@pytest.mark.parametrize("bind", [False, True])
@pytest.mark.parametrize("check_reverse_sync", [False, True])
@pytest.mark.parametrize("use_reverse_sync", [False, True])
@pytest.mark.parametrize(
    "field, value, validation_fails",
    [
        pytest.param("new_language", "English", False, id="change language"),
        pytest.param("openai_model_name", "what", False, id="change model name"),
        pytest.param("openai_model_name", 14, True, id="invalid model name"),
        pytest.param("page", "yo", True, id="invalid page"),
        pytest.param("saved_filenames", ["yoyo"], False, id="saved filenames"),
        pytest.param("openai_client", 5, True, id="invalid openai client"),
    ],
)
def test_session_states_are_synced(
    apptest: AppTest,
    bind: bool,
    field: str,
    value: ty.Any,
    validation_fails: bool,
    check_reverse_sync: bool,
    use_reverse_sync: bool,
    auto_sync: bool,
) -> None:
    """Test `SessionState` is always synced with `st.session_state`."""
    # Reverse sync?
    params = dict(auto_sync=auto_sync, reverse=use_reverse_sync)

    # Create state
    # Context manager ensures state is reset at the end of the test
    with SessionState(**params) as our_state:
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

        # Check if auto_sync is on
        if not auto_sync:
            return  # Tests below are for auto sync functionality

        # Update field
        logger.info(f"(check_reverse_sync={check_reverse_sync}) Updating `{field}` with `{value}`")
        with _VerboseSetItem(depth=6):
            if check_reverse_sync:
                st_state[field] = value
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
