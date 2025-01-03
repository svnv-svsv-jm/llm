__all__ = ["BasicStreamEvent", "ChatMLXEvent", "AgentPayload"]

import typing as ty
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from svsvllm.types import UuidType
from svsvllm.utils import uuid_str
from ._base import BaseModelWithValidation


class BasicStreamEvent(BaseModelWithValidation):
    """Schema for basic event objects.

    These should be simple `dict` objects with a `messages` key.
    """

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        validate_default=True,
    )

    messages: list[BaseMessage] = Field(
        [],
        description="List of exchanged messages.",
        examples=[
            [
                HumanMessage(content="What happens when an unstoppable force meets an immovable object?"),
                AIMessage(
                    content="The unstoppable force will be unable to move the immovable object, as the immovable object is not capable of being moved by an unstoppable force."
                ),
            ]
        ],
    )


class AgentPayload(BaseModelWithValidation):
    """LLM agent payload. This is the schema for event objects returned by a streaming LLM agent."""

    id: UuidType = Field(default_factory=uuid_str, description="ID.")
    name: str = Field("agent", description="Name of the streamer.", examples=["agent"])
    error: ty.Any = Field(None, description="Encountered error(s).")
    interrupts: list[ty.Any] = Field([], description="List of interruptions.")
    result: list[tuple[str, list[BaseMessage]]] = Field(
        [],
        description="List of messages.",
        examples=[
            [
                (
                    "messages",
                    [
                        AIMessage(
                            content="Hi, how can I help you?",
                            additional_kwargs={},
                            response_metadata={},
                            id="run-b65f58d7-0ded-47c7-952e-c1dd0ae244ee-0",
                        )
                    ],
                )
            ]
        ],
    )


class ChatMLXEvent(BaseModelWithValidation):
    """When streaming from a `ChatMLX` chat model, we get:

    ```python
    from langchain_core.messages import AIMessage

    {
        "type": "task_result",
        "timestamp": "2024-12-13T08:29:55.849796+00:00",
        "step": 1,
        "payload": {
            "id": "0a3f2b8c-c72d-b31e-3ce6-3fd0ebaa836c",
            "name": "agent",
            "error": None,
            "result": [
                (
                    "messages",
                    [
                        AIMessage(
                            content="Hi, I am an LLM.",
                            additional_kwargs={},
                            response_metadata={},
                            id="run-b65f58d7-0ded-47c7-952e-c1dd0ae244ee-0",
                        )
                    ],
                )
            ],
            "interrupts": [],
        },
    }
    ```

    This class models this schema.

    It is unclear whether `langchain` can provide this schema directly or not. That would be preferable.
    For the moment, we create it ourselves, with the risk that this may change in the future.
    """

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        validate_default=True,
    )

    messages: list[BaseMessage] = Field(
        [],
        description="List of exchanged messages.",
        examples=[
            [
                HumanMessage(content="What happens when an unstoppable force meets an immovable object?"),
                AIMessage(
                    content="The unstoppable force will be unable to move the immovable object, as the immovable object is not capable of being moved by an unstoppable force."
                ),
            ]
        ],
    )
    type: str = Field(
        "task_result",
        description="Type of event.",
        examples=["task_result"],
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp.",
    )
    step: int = Field(
        0,
        description="Step number.",
        ge=0,
    )
    payload: AgentPayload = Field(
        AgentPayload(),
        description="Payload. This is what the LLM agent is streaming.",
    )
