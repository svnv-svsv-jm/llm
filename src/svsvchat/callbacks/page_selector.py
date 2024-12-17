__all__ = ["PageSelectorCallback"]

import typing as ty
from loguru import logger

from svsvchat.session_state import session_state
from ._base import BaseCallback


class PageSelectorCallback(BaseCallback):
    """Pass this callback to a button, to help change page."""

    def __init__(self, page: str, **kwargs: ty.Any) -> None:
        """
        Args:
            page (str):
                Name of the page where we want to land.
        """
        super().__init__(**kwargs)
        # Attributes
        self.page = page

    def run(self, page: str = None) -> None:
        """Switch page"""
        if page is None:
            page = self.page
        logger.trace(f"Switching to {page}")
        session_state.page = page
        logger.trace(f"Set `page` to {session_state.page}")
