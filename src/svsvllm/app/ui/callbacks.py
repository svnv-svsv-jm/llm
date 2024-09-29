__all__ = ["PageSelectorCallback", "UpdateLanguageCallback"]

import typing as ty
import streamlit as st

from .const import PageNames


class BaseCallback:
    """Base callback."""


class UpdateLanguageCallback(BaseCallback):
    """Callback to update language."""

    def __init__(self, **kwargs: ty.Any) -> None:
        super().__init__(**kwargs)

    def __call__(self) -> None:
        """Update language."""
        st.session_state.language = st.session_state.new_language


class PageSelectorCallback(BaseCallback):
    """Pass this callback to a button, to help change page."""

    def __init__(self, page: str, **kwargs: ty.Any) -> None:
        """
        Args:
            page (str):
                Page name where to land.
        """
        super().__init__(**kwargs)

        # Go to main if unknown page
        if page.lower() not in PageNames.all():
            page = PageNames.MAIN

        # Attribute
        self.page = page

    def __call__(self) -> None:
        """Switch page"""
        st.session_state.page = self.page
