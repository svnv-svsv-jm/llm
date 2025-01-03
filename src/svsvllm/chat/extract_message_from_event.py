__all__ = ["extract_message_from_event"]

import typing as ty
from loguru import logger
from langchain_core.messages import BaseMessage, AIMessage

from svsvllm.exceptions import UnsupportedLLMResponse
from svsvllm.schema import ChatMLXEvent, BasicStreamEvent


def extract_message_from_event(event: dict) -> BaseMessage:
    """Extracts the LLM's response message from its streamed events.

    Args:
        event (dict):
            LLM's streamed event.
            This is expected to be a `dict` with a `messages` key of type `list[BaseMessage]`.

    Raises:
        UnsupportedLLMResponse: If the LLM's message cannot be extracted.

    Returns:
        BaseMessage: LLM's response.
    """
    logger.trace(f"Received event: {event}")
    messages: list[BaseMessage]
    message: BaseMessage

    # Check if `BasicStreamEvent`
    if BasicStreamEvent.is_valid(event):
        ev = BasicStreamEvent(**event)
        messages = ev.messages
        if len(messages) > 0:
            message = messages[-1]
            return message

    # MLX events are a special case
    if ChatMLXEvent.is_valid(event):
        logger.trace("ChatMLXEvent")
        ev = ChatMLXEvent(**event)
        out = ev.payload.result  # pylint: disable=no-member
        logger.trace(f"ChatMLXEvent result: {out}")
        if len(out) < 1:
            return AIMessage(content="")
        name, messages = out[-1]
        logger.trace(f"{name}: {messages}")
        if len(messages) < 1:
            return AIMessage(content="")
        message = messages[-1]
        return message

    raise UnsupportedLLMResponse()
