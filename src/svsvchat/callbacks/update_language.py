__all__ = ["UpdateLanguageCallback"]

import typing as ty
from loguru import logger
import streamlit as st

from svsvchat.session_state import session_state
from ._base import BaseCallback


class UpdateLanguageCallback(BaseCallback):
    """Callback to update language."""

    def run(self) -> None:
        """Update language."""
        logger.trace(f"Updating language: {session_state.language}->{session_state.new_language}")
        st.session_state["language"] = session_state.new_language
