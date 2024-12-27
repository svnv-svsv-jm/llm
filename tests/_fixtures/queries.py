__all__ = ["query"]
import pytest
from unittest.mock import patch, MagicMock
import typing as ty


@pytest.fixture
def query() -> str:
    """Simple LLM query."""
    return "What happens when an unstoppable force meets an immovable object?"
