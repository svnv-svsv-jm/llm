import typing as ty

import pydantic
import streamlit as st
from llama_index.core.llms import ChatMessage

from svsv.types import StateType

T = ty.TypeVar("T")


class JsonSchemaExtra(pydantic.BaseModel):
    """JSON schema extra."""

    is_synced: bool = pydantic.Field(description="Whether this is synced with Streamlit or not.")


class SessionState(pydantic.BaseModel):
    """Session state."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    messages: list[ChatMessage] = pydantic.Field(
        [],
        description="Chat messages.",
        json_schema_extra=JsonSchemaExtra(is_synced=True).model_dump(),
    )

    @property
    def session_state(self) -> StateType:
        """Streamlit session state."""
        return st.session_state


session_state = SessionState()
