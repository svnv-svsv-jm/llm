__all__ = ["check_state_has_cb"]

import pytest
import typing as ty

from svsvchat.session_state import SessionState
from svsvchat.callbacks._base import BaseCallback


@pytest.fixture
def check_state_has_cb(session_state: SessionState) -> ty.Callable[[type[BaseCallback]], bool]:
    """Useful helper function, to check if a certain callback is present in the `session_state` object.
    Callbacks are expected to be registered at `session_state.callbacks`.

    Returns:
        ty.Callable[[type[BaseCallback]], bool]:
            Callable that expects one input argument and returns `True` if a callback of that class exists (in the session state), `False` otherwise.
    """

    def _has_cb(cb_class: type[BaseCallback]) -> bool:
        """Checks if a callback of the specified class exists in the `session_state` object.

        Args:
            cb_class (type[BaseCallback]):
                Class of the callback we want to check the existence of.

        Returns:
            bool: `True` if a callback of that class exists (in the session state), `False` otherwise.
        """
        for _, cb in session_state.callbacks.items():  # pylint: disable=no-member
            if isinstance(cb, cb_class):
                return True
        return False

    return _has_cb
