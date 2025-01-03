import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage

from svsvllm.schema import ChatMLXEvent, AgentPayload
from svsvllm.chat import extract_message_from_event
from svsvllm.exceptions import UnsupportedLLMResponse


class Input(BaseModel):
    """Test input."""

    event: dict = Field(description="Input event to the function under test.")
    message: str = Field(description="Expected extracted message.")
    raises_error: bool = False
    tag: str = ""


CONTENT = "Ciao"
PAYLOAD = AgentPayload(result=[("ok", [AIMessage(content=CONTENT)])])


@pytest.mark.parametrize(
    "arg",
    [
        Input(event=dict(messages=[AIMessage(content=CONTENT)]), message=CONTENT),
        Input(event=dict(payload=PAYLOAD.model_dump()), message=CONTENT),
        Input(
            event=dict(payload=AgentPayload(result=[("ok", [])]).model_dump()),
            message="",
            tag="Empty messages in payload.",
        ),
        Input(
            event=dict(payload=AgentPayload(result=[]).model_dump()),
            message="",
            tag="Empty payload.",
        ),
        Input(event=ChatMLXEvent(payload=PAYLOAD).model_dump(), message=CONTENT),
        Input(event={}, message="", raises_error=True),
    ],
)
def test_extract_message_from_event(arg: Input) -> None:
    """Test `extract_message_from_event`."""
    if arg.raises_error:
        with pytest.raises(UnsupportedLLMResponse):
            extract_message_from_event(arg.event)
        return

    msg = extract_message_from_event(arg.event)
    assert msg.content == arg.message


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
