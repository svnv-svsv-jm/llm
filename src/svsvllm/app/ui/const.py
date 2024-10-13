__all__ = ["DEFAULT_UPLOADED_FILES_DIR", "UPLOADED_FILES_DIR", "PageNames"]

import os
from pathlib import Path

DEFAULT_UPLOADED_FILES_DIR = os.path.join("res", "documents")
Path(DEFAULT_UPLOADED_FILES_DIR).mkdir(parents=True, exist_ok=True)
UPLOADED_FILES_DIR = os.environ.get("UPLOADED_FILES_DIR", DEFAULT_UPLOADED_FILES_DIR)


class PageNames:
    """UI pages' names."""

    SETTINGS = "settings"
    MAIN = "main"

    @classmethod
    def all(cls) -> list[str]:
        """Returns all values."""
        return [cls.SETTINGS, cls.MAIN]
