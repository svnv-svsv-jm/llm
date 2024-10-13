import pytest
from loguru import logger
import typing as ty
import sys, os

import uuid
from svsvllm.loaders import load_documents


def test_load_documents_error() -> None:
    """Test `load_documents` and check it raises an error."""
    with pytest.raises(FileNotFoundError):
        load_documents(f"{uuid.uuid4()}")


def test_load_documents(docs_path: str) -> None:
    """Test `load_documents` and check it raises an error."""
    docs = load_documents(docs_path)
    logger.success(f"Loaded: {docs}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
