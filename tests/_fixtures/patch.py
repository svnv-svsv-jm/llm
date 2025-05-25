import typing as ty
from unittest import mock

import pytest
from llama_index.llms.ollama import Ollama


@pytest.fixture
def stream_chat() -> list[str]:
    """Streaming response to be patch `Ollama` with."""
    return ["hi, ", "how ", "are ", "you?"]


@pytest.fixture
def patch_ollama(stream_chat: list[str]) -> ty.Iterator[mock.MagicMock]:
    """Patch `Ollama` so we can run tests without having to run the server first."""
    ollama = mock.MagicMock(spec=Ollama)
    ollama.stream_chat = mock.MagicMock()
    ollama.stream_chat.return_value = stream_chat
    with (
        mock.patch.object(Ollama, "__new__", return_value=ollama),
        mock.patch.object(Ollama, "__init__", return_value=None),
    ):
        yield ollama
