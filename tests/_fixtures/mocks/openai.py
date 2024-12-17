__all__ = ["mock_openai"]

import pytest
from unittest.mock import MagicMock, patch
import os
import typing as ty
from loguru import logger

from openai import OpenAI


class OpenAIMock(MagicMock):
    """A subclass of MagicMock that is recognized as an instance of `OpenAI`."""

    def __init__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        """Init superclass with `spec=OpenAI`, then create necessary attributes."""
        super().__init__(*args, spec=OpenAI, **kwargs)
        # Mock method that returns LLM's response
        choice = MagicMock()
        choice.message.content = "Hi."
        response = MagicMock()
        response.choices = MagicMock(return_value=[MagicMock(return_value=choice)])
        # Now create the OpenAI mock
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(return_value=response)


@pytest.fixture
def mock_openai() -> ty.Iterator[MagicMock]:
    """Mock `OpenAI` client."""
    client = OpenAIMock()
    assert isinstance(client, OpenAI)
    with patch.object(OpenAI, "__new__", return_value=client) as openaimock:
        yield openaimock
