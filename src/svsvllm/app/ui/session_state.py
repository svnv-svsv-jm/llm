# pylint: disable=no-member,unnecessary-dunder-call,global-variable-not-assigned
__all__ = ["SessionState", "session_state"]

import typing as ty
from loguru import logger
from pydantic import BaseModel, Field, SecretStr, field_validator
import streamlit as st
from streamlit.runtime.state import (
    SafeSessionState,
    SessionStateProxy,
    SessionState as _SessionState,
)
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_huggingface import ChatHuggingFace
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.retrievers import BaseRetriever, RetrieverOutputLike
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

from svsvllm.defaults import EMBEDDING_DEFAULT_MODEL, DEFAULT_LLM, OPENAI_DEFAULT_MODEL
from .callbacks import BaseCallback


_locals: dict[str, ty.Any] = {"__avoid_recursive": False, "_bind": None}


class PatchRecursive:
    """Temporarily deactivate recursion."""

    def __enter__(self) -> "PatchRecursive":
        """Edit value."""
        _locals["__avoid_recursive"] = True
        return self

    def __exit__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        """Edit value."""
        _locals["__avoid_recursive"] = False


class SessionState(BaseModel):
    """Custom Streamlit session state, in order to revise all present keys, validate them, etc."""

    has_chat: bool = Field(
        default=True,
        description="Whether the app has a chatbot or not.",
        validate_default=True,
    )
    language: ty.Literal["English", "Italian"] = Field(
        default="English",
        description="App language.",
        validate_default=True,
    )
    new_language: ty.Literal["English", "Italian"] = Field(
        default="English",
        description="App language we are switching to.",
        validate_default=True,
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="API key for OpenAI.",
        validate_default=True,
    )
    openai_model_name: str = Field(
        default=OPENAI_DEFAULT_MODEL,
        description="OpenAI model name.",
        validate_default=True,
    )
    model_name: str = Field(
        default=DEFAULT_LLM,
        description="HuggingFace model name.",
        validate_default=True,
    )
    embedding_model_name: str = Field(
        default=EMBEDDING_DEFAULT_MODEL,
        description="RAG embedding model name.",
        validate_default=True,
    )
    uploaded_files: list[UploadedFile] | None = Field(
        default=None,
        description="Uploaded files.",
        validate_default=True,
    )
    callbacks: dict[str, BaseCallback] = Field(
        default={},
        description="Uploaded files.",
        validate_default=True,
    )
    saved_filenames: list[str] = Field(
        default=[],
        description="Names of the uploaded files after written to disk.",
        validate_default=True,
    )
    page: ty.Literal["main", "settings"] = Field(
        default="main",
        description="Page we are displaying.",
        validate_default=True,
    )
    openai_client: OpenAI | None = Field(
        default=None,
        description="OpenAI client.",
        validate_default=True,
    )
    hf_model: AutoModelForCausalLM | None = Field(
        default=None,
        description="HuggingFace model.",
        validate_default=True,
    )
    tokenizer: AutoTokenizer | None = Field(
        default=None,
        description="HuggingFace tokenizer.",
        validate_default=True,
    )
    db: FAISS | None = Field(
        default=None,
        description="Document database.",
        validate_default=True,
    )
    retriever: BaseRetriever | None = Field(
        default=None,
        description="Retriever object created from databse.",
        validate_default=True,
    )
    history_aware_retriever: RetrieverOutputLike | None = Field(
        default=None,
        description="History aware retriever.",
        validate_default=True,
    )
    chat_model: ChatHuggingFace | None = Field(
        default=None,
        description="Open source HuggingFace model LangChain wrapper.",
        validate_default=True,
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
        # Immediately sync
        self.sync()
        # Patch
        self.patch(st.session_state)

    @property
    def _bind(self) -> SessionStateProxy | SafeSessionState | None:
        """Access the bound sesstion state."""
        r: SessionStateProxy | SafeSessionState | None = _locals["_bind"]
        if r is not None:
            assert isinstance(r, (_SessionState, SessionStateProxy, SafeSessionState))
        return r

    @_bind.setter
    def _bind(self, value: SessionStateProxy | SafeSessionState) -> None:
        """Set the bound object."""
        _locals["_bind"] = value

    @property
    def __avoid_recursive(self) -> bool:
        """Access the bound sesstion state."""
        r: bool = _locals["__avoid_recursive"]
        return r

    @__avoid_recursive.setter
    def __avoid_recursive(self, value: bool) -> None:
        """Set the bound object."""
        _locals["__avoid_recursive"] = value

    def sync(self) -> None:
        """Sync states."""
        logger.debug(f"Syncing")
        for key, val in self.to_dict().items():
            logger.trace(f"Syncing: {key}")
            self.session_state[key] = val

    def bind(self, session_state: SessionStateProxy | SafeSessionState) -> None:
        """Bind this model to Streamlit's session state."""
        logger.trace(f"Binding to {type(session_state)}")
        self._bind = self.patch(session_state)
        self.sync()

    def unbind(self) -> None:
        """Unbind."""
        logger.trace("Unbinding")
        self._bind = None

    def patch(
        self,
        state: SessionStateProxy | SafeSessionState | None = None,
    ) -> SessionStateProxy | SafeSessionState:
        """We patch the original Streamlit state so that when that one is used, it automatically syncs with this class."""
        if state is None:
            state = self.session_state
        logger.trace("Patching `__setitem__` and `__setattr__`")
        # Patch
        __setitem__original = state.__class__.__setitem__
        __setattr__original = state.__class__.__setattr__

        def patched_setitem(
            obj: SessionStateProxy | SafeSessionState,
            key: ty.Any,
            value: ty.Any,
        ) -> None:
            # Call the original method
            __setitem__original(obj, key, value)  # type: ignore
            # Call custom function
            with PatchRecursive():
                self.__setitem__(key, value)

        def patched_setattr(
            obj: SessionStateProxy | SafeSessionState,
            key: ty.Any,
            value: ty.Any,
        ) -> None:
            # Call the original method
            __setattr__original(obj, key, value)  # type: ignore
            # Call custom function
            with PatchRecursive():
                self.__setattr__(key, value)

        # Replace the original method with the patched one
        state.__class__.__setitem__ = patched_setitem  # type: ignore
        state.__class__.__setattr__ = patched_setattr  # type: ignore
        logger.trace("Patched `__setitem__` and `__setattr__`")
        return state

    @property
    def session_state(self) -> SessionStateProxy | SafeSessionState:
        """Streamlit session state."""
        if self._bind is not None:
            logger.trace("Session state is bounded explicitly.")
            state = self._bind
        else:
            logger.trace("Session state is not bounded explicitly")
            state = st.session_state
        # Return
        return state

    def __len__(self) -> int:
        """Number of user state and keyed widget values in session_state."""
        return self.session_state.__len__()

    def __getitem__(self, key: str) -> ty.Any:
        """Return the state or widget value with the given key."""
        return self.session_state.__getitem__(key)

    def __setitem__(self, key: str, value: ty.Any) -> None:
        """Set the value of the given key."""
        self.__setattr__(key, value)

    def __setattr__(self, key: str, value: ty.Any) -> None:
        """Set the value of the given key."""
        super().__setattr__(key, value)  # type: ignore
        if self.__avoid_recursive:
            return
        self.session_state.__setattr__(key, value)

    def to_dict(self) -> dict[str, ty.Any]:
        """Dump model."""
        return self.model_dump()


session_state = SessionState()
