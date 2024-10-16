__all__ = ["get_and_maybe_init_session_state"]

import typing as ty
from loguru import logger
import streamlit as st


def get_and_maybe_init_session_state(key: str, initial_value: ty.Any = None) -> ty.Any:
    """Gets element from session state, initializing if it does not exist.

    Args:
        key (str):
            Name of the element in the session state.
            The element will be retrieved as `st.session_state.get(key)`.

        initial_value (ty.Any):
            If `st.session_state[key]` does not exist, it will be initialized with this value.

    Returns:
        ty.Any: The element that is `st.session_state.get(key)`.
    """
    logger.trace(f"Getting '{key}' in session state")
    if st.session_state.get(key, None) is None and initial_value is not None:
        logger.trace(f"Creating '{key}' in session state")
        st.session_state[key] = initial_value
    value = st.session_state.get(key)
    return value
