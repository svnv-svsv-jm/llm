# pylint: disable=no-member,unnecessary-dunder-call,global-variable-not-assigned
__all__ = ["SessionState"]

import typing as ty
from loguru import logger
from pydantic import BaseModel, Field, SecretStr, field_validator
import streamlit as st
from streamlit.runtime.state import (
    SafeSessionState,
    SessionStateProxy,
    SessionState as StreamlitSessionState,
)
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_huggingface import ChatHuggingFace
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.retrievers import BaseRetriever, RetrieverOutputLike
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

from svsvllm.utils.singleton import Singleton
from svsvllm.defaults import EMBEDDING_DEFAULT_MODEL, DEFAULT_LLM, OPENAI_DEFAULT_MODEL
from .callbacks import BaseCallback


StateType = StreamlitSessionState | SessionStateProxy | SafeSessionState
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


class _SessionState(BaseModel):
    """Custom Streamlit session state, in order to revise all present keys, validate them, etc.

    Instances of this class are fully synced with Streamlit's session state. This means that changes to `import streamlit as st; st.session_state` will be reflected in instances of this class, and reverse.
    Thus, please ensure only one instance of this class exists at a time.
    """

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
    callbacks: dict[str, BaseCallback] | None = Field(
        default=None,
        description="Uploaded files.",
        validate_default=True,
    )
    saved_filenames: list[str] | None = Field(
        default=None,
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
    def _bind(self) -> StateType | None:
        """Access the bound sesstion state."""
        r: StateType | None = _locals["_bind"]
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

    @property
    def session_state(self) -> StateType:
        """Streamlit session state."""
        if self._bind is not None:
            logger.trace("Session state is bounded explicitly.")
            state = self._bind
        else:
            logger.trace("Session state is not bounded explicitly")
            state = st.session_state
        # Return
        return state

    @logger.catch(AttributeError, message="Failed to sync with Streamlit session state.")
    def _sync(self, from_state: ty.Mapping[str, ty.Any], to_state: ty.Mapping[str, ty.Any]) -> None:
        logger.debug(f"Syncing from {type(from_state)} to {type(to_state)}")
        for key, val in from_state.items():
            logger.trace(f"Syncing: {key}")
            try:
                to_state[key] = val  # type: ignore
            except (KeyError, AttributeError):
                logger.debug(f"Failed to sync `{key}`")
        logger.debug(f"Synced from {type(from_state)} to {type(to_state)}")

    def sync(self, reverse: bool = False) -> None:
        """Sync states.

        Args:
            reverse (bool):
                If `True`, try to sync from `st.session_state` to here. If this raises `AttributeError`, this means the state is empty, so we sync from here to there.
        """
        if reverse:
            self._sync(from_state=self.session_state, to_state=self)  # type: ignore
        self._sync(from_state=self.to_dict(), to_state=self.session_state)  # type: ignore

    def bind(self, session_state: StateType) -> None:
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
        state: StateType | None = None,
    ) -> StateType:
        """We patch the original Streamlit state so that when that one is used, it automatically syncs with this class."""
        if state is None:
            state = self.session_state
        logger.trace("Patching `__setitem__` and `__setattr__`")
        # Patch
        __setitem__original = state.__class__.__setitem__
        __setattr__original = state.__class__.__setattr__

        def patched_setitem(
            obj: StateType,
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

    def __len__(self) -> int:
        """Number of user state and keyed widget values in session_state."""
        return self.session_state.__len__()

    def __getitem__(self, key: str | int) -> ty.Any:
        """Return the state or widget value with the given key."""
        return self.session_state.__getitem__(key)  # type: ignore

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


class SessionState(Singleton):
    """Custom Streamlit session state, in order to revise all present keys, validate them, etc.

    Instances of this class are fully synced with Streamlit's session state. This means that changes to `import streamlit as st; st.session_state` will be reflected in instances of this class, and reverse.

    This class is a singleton, so only one instance can be created at a time.

    Technically, this is a wrapper around the `pydantic` model :class:`_SessionState`.
    """

    def __init__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        """Constructor.

        Args:
            **kwargs (Any):
                See :class:`_SessionState`.
        """
        self.state = _SessionState(**kwargs)

    @property
    def state(self) -> _SessionState:
        """Return the state of the session."""
        return self._state

    @state.setter
    def state(self, state: _SessionState) -> None:
        """Set the state of the session."""
        self._state = state

    @property
    def session_state(self) -> StateType:
        """Streamlit session state."""
        return self.state.session_state

    def bind(self, session_state: StateType) -> None:
        """Bind this model to Streamlit's session state."""
        self.state.bind(session_state)

    def unbind(self) -> None:
        """Unbind."""
        logger.trace(f"Unbinding {self}")
        self.state.unbind()

    def sync(self) -> None:
        """Sync states."""
        self.state.sync()

    def to_dict(self) -> dict[str, ty.Any]:
        """Dump model."""
        return self.state.to_dict()

    def __len__(self) -> int:
        """Number of user state and keyed widget values in session_state."""
        return len(self.state)

    def __getitem__(self, key: str) -> ty.Any:
        """Return the state or widget value with the given key."""
        return self.state[key]

    def __setitem__(self, key: str, value: ty.Any) -> None:
        """Set the value of the given key."""
        self.state[key] = value

    def __getattr__(self, key: str) -> ty.Any:
        """get the value from the wrapped state."""
        if key not in ["_state", "state"]:
            return getattr(self._state, key)
        # return super().__getattr__(key)

    def __setattr__(self, key: str, value: ty.Any) -> None:
        """Set the value of the given key, but set it on the wrapped satte."""
        if key not in ["_state", "state"]:
            setattr(self._state, key, value)
        else:
            super().__setattr__(key, value)
