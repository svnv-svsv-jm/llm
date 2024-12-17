__all__ = [
    "HUGGINFACE_TOKEN_KEY",
    "ENV_PREFIX",
    "LOG_LEVEL_KEY",
    "DEFAULT_UPLOADED_FILES_DIR",
    "Q_SYSTEM_PROMPT",
    "PageNames",
]

import os
from pathlib import Path

HUGGINFACE_TOKEN_KEY = "HF_TOKEN"

# Environment prefix for settings
ENV_PREFIX = "SVSVLLM_"

# Logging level key
LOG_LEVEL_KEY = "LOG_LEVEL"

# Default location for documents
DEFAULT_UPLOADED_FILES_DIR = os.path.join(".rag")
Path(DEFAULT_UPLOADED_FILES_DIR).mkdir(parents=True, exist_ok=True)


Q_SYSTEM_PROMPT = "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."


class PageNames:
    """UI pages' names."""

    SETTINGS = "settings"
    MAIN = "main"

    @classmethod
    def all(cls) -> list[str]:
        """Returns all values."""
        return [cls.SETTINGS, cls.MAIN]
