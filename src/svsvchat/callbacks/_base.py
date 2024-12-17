from loguru import logger
import streamlit as st

from svsvchat.session_state import session_state


class BaseCallback:
    """Base callback."""

    def __init__(self, name: str) -> None:
        self._name = name
        st.session_state.setdefault("callbacks", session_state.callbacks)
        logger.trace(f"Adding {self.name} to session state.")
        session_state.callbacks[name] = self
        logger.trace(f"Added {self.name} to session state: {st.session_state}")

    @property
    def name(self) -> str:
        """Name of the callback."""
        nature = self.__class__.__name__
        name = self._name
        return f"{nature}({name})"

    def __repr__(self) -> str:
        return self.name

    def __call__(self) -> None:
        """Main caller."""
        logger.trace(f"Running {self.name}")
        self.run()

    def run(self) -> None:
        """Subclass method."""
        raise NotImplementedError()
