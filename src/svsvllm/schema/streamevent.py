__all__ = ["ChatMLXEvent", "AgentPayload"]

import typing as ty
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from langchain_core.messages import AIMessage, BaseMessage

from svsvllm.types import UuidType


class AgentPayload(BaseModel):
    """LLM agent payload. This is the schema for event objects returned by a streaming LLM agent."""

    id: UuidType = Field(description="ID.")
    name: str = Field(description="Name of the streamer.", examples=["agent"])
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


class ChatMLXEvent(BaseModel):
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

    type: str = Field("task_result", description="Type of event.", examples=["task_result"])
    timestamp: datetime = Field(description="Timestamp.")
    step: int = Field(0, description="Step.", ge=0)
    payload: AgentPayload = Field(description="Payload. This is what the LLM agent is streaming.")

    @classmethod
    def is_valid(cls, obj: dict) -> bool:
        """`True` if input complies to schema."""
        try:
            cls.model_validate(obj)
            return True
        except:
            return False
