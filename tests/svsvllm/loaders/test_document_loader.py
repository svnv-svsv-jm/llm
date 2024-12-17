import pytest
from loguru import logger
import typing as ty
import sys, os

import uuid
from langchain_core.documents import Document

from svsvllm.loaders import load_documents


def test_load_documents_error() -> None:
    """Test `load_documents` and check it raises an error."""
    with pytest.raises(FileNotFoundError):
        load_documents(f"{uuid.uuid4()}")


def test_load_documents(documents: list[Document]) -> None:
    """Test `load_documents` and check it raises an error."""
    logger.success(f"Loaded: {documents}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
