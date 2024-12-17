__all__ = ["mock_agent_stream"]

import pytest
from unittest.mock import MagicMock, patch
import os
import typing as ty
from loguru import logger

from langchain_core.messages import AIMessage
from langgraph.graph.graph import CompiledGraph


@pytest.fixture
def mock_agent_stream(request: pytest.FixtureRequest) -> ty.Iterator[MagicMock]:
    """Mock agent's `CompiledGraph.stream()` method.
    We create a mock `dict` object with the `"messages"` key having a `list[AIMessage]` value.
    """
    msg = f'{getattr(request, "param", "This is a mocked message.")}'
    # Create an object to stream
    stream = MagicMock(return_value={"messages": [AIMessage(content=msg)]})
    # Patch and pass a list `side_effect` to simulate the `yield` effect (streaming)
    with patch.object(CompiledGraph, "stream", side_effect=[stream] * 5) as mock_stream:
        yield mock_stream
