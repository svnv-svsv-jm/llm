__all__ = [
    "DEFAULT_UPLOADED_FILES_DIR",
    "UPLOADED_FILES_DIR",
    "Q_SYSTEM_PROMPT",
    "OPEN_SOURCE_MODELS_SUPPORTED",
    "PageNames",
]

import os
from pathlib import Path

DEFAULT_UPLOADED_FILES_DIR = os.path.join(".rag")
Path(DEFAULT_UPLOADED_FILES_DIR).mkdir(parents=True, exist_ok=True)
UPLOADED_FILES_DIR = os.environ.get("UPLOADED_FILES_DIR", DEFAULT_UPLOADED_FILES_DIR)

Q_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

OPEN_SOURCE_MODELS_SUPPORTED = False


class PageNames:
    """UI pages' names."""

    SETTINGS = "settings"
    MAIN = "main"

    @classmethod
    def all(cls) -> list[str]:
        """Returns all values."""
        return [cls.SETTINGS, cls.MAIN]
