# pylint: disable=no-member
__all__ = ["SessionState"]

import typing as ty
from loguru import logger
from pydantic import BaseModel, Field, SecretStr, field_validator
import streamlit as st
from streamlit.runtime.state.safe_session_state import SafeSessionState
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_huggingface import ChatHuggingFace
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.retrievers import BaseRetriever, RetrieverOutputLike
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

from svsvllm.defaults import EMBEDDING_DEFAULT_MODEL, DEFAULT_LLM, OPENAI_DEFAULT_MODEL
from .callbacks import BaseCallback


class SessionState(BaseModel):
    """Custom Streamlit session state, in order to revise all present keys, validate them, etc."""

    has_chat: bool = Field(
        default=True,
        description="Whether the app has a chatbot or not.",
        validate_default=False,
    )
    language: ty.Literal["English", "Italian"] = Field(
        default="English",
        description="App language.",
        validate_default=False,
    )
    new_language: ty.Literal["English", "Italian"] = Field(
        default="English",
        description="App language we are switching to.",
        validate_default=False,
    )
    openai_api_key: SecretStr = Field(
        default=None,
        description="API key for OpenAI.",
        validate_default=False,
    )
    openai_model_name: str = Field(
        default=OPENAI_DEFAULT_MODEL,
        description="OpenAI model name.",
        validate_default=False,
    )
    model_name: str = Field(
        default=DEFAULT_LLM,
        description="HuggingFace model name.",
        validate_default=False,
    )
    embedding_model_name: str = Field(
        default=EMBEDDING_DEFAULT_MODEL,
        description="RAG embedding model name.",
        validate_default=False,
    )
    uploaded_files: list[UploadedFile] = Field(
        default=None,
        description="Uploaded files.",
        validate_default=False,
    )
    callbacks: dict[str, BaseCallback] = Field(
        default={},
        description="Uploaded files.",
        validate_default=False,
    )
    saved_filenames: list[str] = Field(
        default=[],
        description="Names of the uploaded files after written to disk.",
        validate_default=False,
    )
    page: ty.Literal["main", "settings"] = Field(
        default="main",
        description="Page we are displaying.",
        validate_default=False,
    )
    openai_client: OpenAI = Field(
        default=None,
        description="OpenAI client.",
        validate_default=False,
    )
    hf_model: AutoModelForCausalLM = Field(
        default=None,
        description="HuggingFace model.",
        validate_default=False,
    )
    tokenizer: AutoTokenizer = Field(
        default=None,
        description="HuggingFace tokenizer.",
        validate_default=False,
    )
    db: FAISS = Field(
        default=None,
        description="Document database.",
        validate_default=False,
    )
    retriever: BaseRetriever = Field(
        default=None,
        description="Retriever object created from databse.",
        validate_default=False,
    )
    history_aware_retriever: RetrieverOutputLike = Field(
        default=None,
        description="History aware retriever.",
        validate_default=False,
    )
    chat_model: ChatHuggingFace = Field(
        default=None,
        description="Open source HuggingFace model LangChain wrapper.",
        validate_default=False,
    )

    class Config:
        """Lets you re-run the validation functions on attribute assignments."""

        validate_assignment = True
        arbitrary_types_allowed = True

    @classmethod
    @field_validator("uploaded_files")
    def _uploaded_files(cls, value: list[UploadedFile]) -> list[UploadedFile]:
        """Validate `uploaded_files`."""
        assert isinstance(value, list)
        for v in value:
            assert isinstance(v, UploadedFile)
        return value

    # -------------
    # Custom implementation
    # -------------

    def __init__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        super().__init__(*args, **kwargs)

        # Initialize bound state
        self._bind: ty.MutableMapping | SafeSessionState | None = None

        # Immediately sync
        self.sync()

    def sync(self) -> None:
        """Sync states."""
        for key, val in self.to_dict().items():
            logger.trace(f"Syncing: {key}")
            self.session_state[key] = val

    def bind(self, session_state: ty.MutableMapping | SafeSessionState) -> None:
        """Bind this model to Streamlit's session state."""
        logger.trace(f"Binding to {type(session_state)}")
        self._bind = session_state
        self.sync()

    def unbind(self) -> None:
        """Unbind."""
        self._bind = None

    @property
    def session_state(self) -> ty.MutableMapping | SafeSessionState:
        """Streamlit session state."""
        if self._bind is not None:
            logger.trace("Session state is bounded explicitly.")
            return self._bind
        logger.trace("Session state is not bounded explicitly")
        return st.session_state

    def __len__(self) -> int:
        """Number of user state and keyed widget values in session_state."""
        return self.session_state.__len__()

    def __getitem__(self, key: str) -> ty.Any:
        """Return the state or widget value with the given key."""
        logger.trace(f"Getting `{key}`")
        return self.session_state.__getitem__(key)

    def __setitem__(self, key: str, value: ty.Any) -> None:
        """Set the value of the given key."""
        logger.trace(f"Setting `{key}` with `{value}`")
        try:
            super().__setitem__(key, value)  # type: ignore
        except:
            logger.trace("Superclass has no `__setitem__()` method.")
            pass
        self.session_state[key] = value

    def __setattr__(self, key: str, value: ty.Any) -> None:
        """Set the value of the given key."""
        logger.trace(f"Setting `{key}` with `{value}`")
        super().__setattr__(key, value)  # type: ignore
        self.session_state[key] = value

    def __getattr__(self, key: str) -> ty.Any:
        """Allow key access."""
        logger.trace(f"Getting `{key}`")
        try:
            return self[key]
        except KeyError as ex:
            raise AttributeError("Invalid key.") from ex

    def to_dict(self) -> dict[str, ty.Any]:
        """Dump model."""
        return self.model_dump()
