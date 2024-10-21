# pylint: disable=no-member,unnecessary-dunder-call,global-variable-not-assigned
__all__ = ["SessionState", "set_state", "get_state", "get_and_maybe_init_session_state"]

import typing as ty
from loguru import logger
from pydantic import BaseModel, Field, SecretStr, field_validator
from io import BytesIO
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


StateType = StreamlitSessionState | SessionStateProxy | SafeSessionState
_locals: dict[str, ty.Any] = {"__avoid_recursive": False, "_bind": None}

DEFAULT_REVERSE_SYNC = False


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
    uploaded_files: list[UploadedFile | BytesIO] | None = Field(
        default=None,
        description="Uploaded files.",
        validate_default=True,
    )
    callbacks: dict[str, ty.Callable] | None = Field(
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

    def _sync(self, from_state: ty.Mapping[str, ty.Any], to_state: ty.Mapping[str, ty.Any]) -> None:
        """Syncer.

        Args:
            from_state (ty.Mapping[str, ty.Any]):
                State that is used as source of truth.

            to_state (ty.Mapping[str, ty.Any]):
                State that we sync with the source of truth.
        """
        logger.debug(f"Syncing from {type(from_state)} to {type(to_state)}")
        # Get items
        try:
            items = from_state.items()
        except Exception as ex:
            logger.debug(ex)
            return
        # Sync
        for key, val in items:
            logger.trace(f"Syncing: {key}")
            try:
                to_state[key] = val  # type: ignore
            except (KeyError, AttributeError):
                logger.debug(f"Failed to sync `{key}`")
        logger.debug(f"Synced from {type(from_state)} to {type(to_state)}")

    def _sync_direct(self) -> None:
        """Direct synchronization."""
        logger.trace("Direct synchronization")
        self._sync(from_state=self.to_dict(), to_state=self.session_state)  # type: ignore

    def _sync_reverse(self) -> None:
        """Reverse synchronization."""
        logger.trace("Reverse synchronization")
        self._sync(from_state=self.session_state, to_state=self)  # type: ignore

    def sync(self, reverse: bool = DEFAULT_REVERSE_SYNC) -> None:
        """Sync states.

        Args:
            reverse (bool):
                If `True`, try to sync from `st.session_state` to here. If this raises `AttributeError`, this means the state is empty, so we sync from here to there.
        """
        if reverse:
            self._sync_reverse()
        self._sync_direct()

    def bind(self, session_state: StateType | None) -> None:
        """Bind this model to Streamlit's session state."""
        if session_state is None:
            logger.trace("Nothing to bind to.")
            return
        logger.trace(f"Binding to {type(session_state)}")
        self._bind = self.patch(session_state)
        self.sync()

    def unbind(self) -> None:
        """Unbind."""
        logger.trace("Unbinding")
        self._bind = None
        self.sync()

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
            # Call custom function first: this runs the pydantic validation before setting the value in the original streamlit state
            with PatchRecursive():
                self.__setitem__(key, value)
            # Call the original method
            __setitem__original(obj, key, value)  # type: ignore
            assert obj[key] == value
            assert self[key] == value

        def patched_setattr(
            obj: StateType,
            key: ty.Any,
            value: ty.Any,
        ) -> None:
            # Call custom function first: this runs the pydantic validation before setting the value in the original streamlit state
            with PatchRecursive():
                self.__setattr__(key, value)
            # Call the original method
            __setattr__original(obj, key, value)  # type: ignore
            assert obj[key] == value
            assert self[key] == value

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


class SessionState(metaclass=Singleton):
    """Custom Streamlit session state, in order to revise all present keys, validate them, etc.

    Instances of this class are fully synced with Streamlit's session state. This means that changes to `import streamlit as st; st.session_state` will be reflected in instances of this class, and reverse.

    This class is a singleton, so only one instance can be created at a time.

    Technically, this is a wrapper around the `pydantic` model :class:`_SessionState`.
    """

    def __init__(
        self,
        reverse: bool,
        state: StateType | None = None,
        **kwargs: ty.Any,
    ) -> None:
        """Constructor.

        Args:
            reverse (bool):
                If `True`, try to sync from `st.session_state` to here. If this raises `AttributeError`, this means the state is empty, so we sync from here to there.

            state (StateType | None):
                Session state to patch.

            **kwargs (Any):
                See :class:`_SessionState`.
        """
        logger.debug(f"Creating session state with:\n\treverse={reverse}")
        self.state = _SessionState(**kwargs)
        self.bind(state)
        self.sync(reverse=reverse)
        self.patch(state)

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

    def sync(self, reverse: bool = DEFAULT_REVERSE_SYNC) -> None:
        """Sync states.

        Args:
            reverse (bool):
                If `True`, try to sync from `st.session_state` to here. If this raises `AttributeError`, this means the state is empty, so we sync from here to there.
        """
        logger.debug(f"Syncing... (reverse={reverse})")
        self.state.sync(reverse=reverse)

    def patch(self, state: StateType | None = None) -> StateType:
        """We patch the original Streamlit state so that when that one is used, it automatically syncs with this class."""
        return self.state.patch(state)

    def bind(self, session_state: StateType | None) -> None:
        """Bind this model to Streamlit's session state."""
        self.state.bind(session_state)

    def unbind(self) -> None:
        """Unbind."""
        logger.trace(f"Unbinding {self}")
        self.state.unbind()

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


def set_state(key: str, value: ty.Any) -> None:
    """Sets an element in the session state, leveraging our custom `pydantic` model for it.

    Args:
        key (str):
            Name of the element in the session state.
            The element will be set as `st.session_state[key] = value`.

        value (ty.Any):
            Value to set. This will be validated using our custom `pydantic` model for the session state.
    """
    # Validate item
    _SessionState(**{key: value})
    # All good, set it
    st.session_state[key] = value


def get_state(key: str) -> ty.Any:
    """Gets element from session state, initializing if it does not exist, leveraging our custom `pydantic` model for it.

    Args:
        key (str):
            Name of the element in the session state.
            The element will be retrieved as `st.session_state.get(key)`.

    Returns:
        ty.Any: The element that is `st.session_state.get(key)`.
    """
    return get_and_maybe_init_session_state(key)


def get_and_maybe_init_session_state(key: str, initial_value: ty.Any = None) -> ty.Any:
    """Gets element from session state, initializing if it does not exist, leveraging our custom `pydantic` model for it.

    Args:
        key (str):
            Name of the element in the session state.
            The element will be retrieved as `st.session_state.get(key)`.

        initial_value (ty.Any):
            If `st.session_state[key]` does not exist, it will be initialized with this value.
            If this is not provided, the default value will be taken from `_SessionState`.

    Returns:
        ty.Any: The element that is `st.session_state.get(key)`.
    """
    logger.trace(f"Getting '{key}' in session state")
    if st.session_state.get(key, None) is None:
        logger.trace(f"Creating '{key}' in session state")
        if initial_value is None:
            initial_value = _SessionState()[key]
        st.session_state[key] = initial_value
    value = st.session_state.get(key)
    return value
