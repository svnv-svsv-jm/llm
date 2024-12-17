__all__ = ["select_init_msg"]

from loguru import logger

from svsvchat.settings import settings
from svsvchat.session_state import session_state


def select_init_msg() -> str:
    """Selects initial message based on language."""
    logger.trace(f"Language: {session_state.language}")
    if session_state.language == "Italian":
        start_msg: str = settings.start_message_it
    else:
        start_msg = settings.start_message_en
    logger.trace(f"Initial message: {start_msg}")
    return start_msg
