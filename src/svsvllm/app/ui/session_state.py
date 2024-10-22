# pylint: disable=no-member,unnecessary-dunder-call,global-variable-not-assigned
__all__ = ["SessionState"]

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
from langchain_core.messages import BaseMessage
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

from svsvllm.utils.singleton import Singleton
from svsvllm.defaults import EMBEDDING_DEFAULT_MODEL, DEFAULT_LLM, OPENAI_DEFAULT_MODEL


StateType = StreamlitSessionState | SessionStateProxy | SafeSessionState
_locals: dict[str, ty.Any] = {
    "_avoid_recursive": False,
    "_bind": None,
    "_verbose": False,
    "_depth": 0,
}
_cache = {}

DEFAULT_REVERSE_SYNC = True


class _PatchRecursive:
    """Temporarily deactivate recursion."""

    def __enter__(self) -> "_PatchRecursive":
        """Edit value."""
        _locals["_avoid_recursive"] = True
        return self

    def __exit__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        """Edit value."""
        _locals["_avoid_recursive"] = False


class _VerboseSetItem:
    """Verbose `setattr` and `setitem` method."""

    def __init__(self, depth: int = 0) -> None:
        """
        Args:
            depth (int):
                Input to `logger.opt(depth=depth)`.
        """
        self.depth = depth

    def __enter__(self) -> "_VerboseSetItem":
        """Edit value."""
        _locals["_verbose"] = True
        _locals["_depth"] = self.depth
        return self

    def __exit__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        """Edit value."""
        _locals["_verbose"] = False
        _locals["_depth"] = 0


class _SessionState(BaseModel):
    """Custom Streamlit session state, in order to revise all present keys, validate them, etc.

    Instances of this class are fully synced with Streamlit's session state. This means that changes to `import streamlit as st; st.session_state` will be reflected in instances of this class, and reverse.
    Thus, please ensure only one instance of this class exists at a time.
    """

    auto_sync: bool = Field(
        default=False,
        description="Whether to automatically sync with Streamlit's state.",
        validate_default=True,
    )
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
    uploaded_files: list[UploadedFile | BytesIO] = Field(
        default=[],
        description="Uploaded files.",
        validate_default=True,
    )
    callbacks: dict[str, ty.Callable] = Field(
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
    messages: list[BaseMessage] = Field(
        default=[],
        description="History of messages.",
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
    def _bind(self, value: StateType | None) -> None:
        """Set the bound object."""
        _locals["_bind"] = value

    @property
    def _avoid_recursive(self) -> bool:
        """Access the bound sesstion state."""
        r: bool = _locals["_avoid_recursive"]
        return r

    @property
    def _verbose_item_set(self) -> bool:
        """Whether to log on item/attribute being set."""
        r: bool = _locals["_verbose"]
        return r

    @property
    def _logging_depth(self) -> bool:
        """Depth when logging."""
        r: int = _locals["_depth"]
        return r

    @property
    def session_state(self) -> StateType:
        """Streamlit session state."""
        if self._bind is not None:
            state = self._bind
        else:
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
        if not self.auto_sync:
            logger.debug("Sync is disabled.")
            return
        logger.debug(f"Syncing from {type(from_state)} to {type(to_state)}")
        # Get items
        try:
            items = from_state.items()
        except Exception as ex:
            logger.debug(ex)
            return
        # Sync
        for key, val in items:
            if "__setattr__" in key.lower():
                continue
            logger.trace(f"Syncing: {key}")
            try:
                to_state[key] = val  # type: ignore
            except (KeyError, AttributeError):
                logger.debug(f"Failed to sync `{key}`")
        logger.debug(f"Synced from {type(from_state)} to {type(to_state)}")

    def _sync_direct(self) -> None:
        """Direct synchronization."""
        logger.debug("Direct synchronization")
        self._sync(from_state=self.to_dict(), to_state=self.session_state)  # type: ignore

    def _sync_reverse(self) -> None:
        """Reverse synchronization."""
        logger.debug("Reverse synchronization")
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

    def bind(self, session_state: StateType | None = None) -> None:
        """Bind this model to Streamlit's session state."""
        logger.trace(f"Binding to {type(session_state)}")
        self._bind = self.patch(session_state)
        self.sync()

    def unbind(self) -> None:
        """Unbind."""
        logger.trace("Unbinding")
        self._bind = None
        self.patch(st.session_state)
        self.sync()

    def patch(
        self,
        state: StateType | None = None,
    ) -> StateType:
        """We patch the original Streamlit state so that when that one is used, it automatically syncs with this class."""
        # Get state to patch
        if state is None:
            state = self.session_state

        # # Do nothing if auto-sync is disabled
        # if not self.auto_sync:
        #     logger.trace("Auto sync is disabled.")
        #     return state

        logger.trace("Patching `__setitem__` and `__setattr__`")

        # By using a local cache, we understand if the state coming in is different or the same as before
        _last_state = _cache.get("last", None)
        # If there is not last state, this is the first time this runs, so save everything
        if _last_state is None:
            _cache["last"] = state
            _cache["__setitem__"] = state.__class__.__setitem__
            _cache["__setattr__"] = state.__class__.__setattr__
        # If they are not the same, we save the references to its methods
        if state is not _cache["last"]:
            _cache["__setitem__"] = state.__class__.__setitem__
            _cache["__setattr__"] = state.__class__.__setattr__
        # Now we get them. We explictly use the `[]` access to let an error be raised if these keys do not exist
        _cache["last"] = state
        __setitem__original = _cache["__setitem__"]
        __setattr__original = _cache["__setattr__"]

        __setattr_call = super().__setattr__

        # Patch
        def patched_setitem(
            obj: StateType,
            key: ty.Any,
            value: ty.Any,
        ) -> None:
            # Call custom function first: this runs the pydantic validation before setting the value in the original streamlit state
            with _PatchRecursive():
                __setattr_call(key, value)
            # Call the original method
            __setitem__original(obj, key, value)  # type: ignore

        def patched_setattr(
            obj: StateType,
            key: ty.Any,
            value: ty.Any,
        ) -> None:
            # Call custom function first: this runs the pydantic validation before setting the value in the original streamlit state
            with _PatchRecursive():
                __setattr_call(key, value)
            # Call the original method
            __setattr__original(obj, key, value)  # type: ignore

        # Replace the original method with the patched one
        state.__class__.__setitem__ = patched_setitem  # type: ignore
        state.__class__.__setattr__ = patched_setattr  # type: ignore
        logger.trace("Patched `__setitem__` and `__setattr__`")
        return state

    def __len__(self) -> int:
        """Number of user state and keyed widget values in session_state."""
        return len(self.session_state)

    def __get(self, key: str) -> ty.Any:
        default = self.__dict__.get(key, None)
        try:
            value = self.session_state[key]
        except KeyError:
            value = default
        # Validate value
        try:
            _SessionState(**{key: value})
        except Exception:
            # If validation fails, choose the default
            value = default
        self.session_state[key] = value
        return value

    def __getitem__(self, key: str | int) -> ty.Any:
        """Return the state or widget value with the given key."""
        return self.__get(key)

    def __getattr__(self, key: str) -> ty.Any:
        """Return the state or widget value with the given key."""
        return self.__get(key)

    def get(self, key: str) -> ty.Any:
        """Getter."""
        return self.__get(key)

    def __set__(self, key: str, value: ty.Any) -> None:
        """Set the value of the given key."""
        # Set by calling pydantic `__setattr__` to run validations
        super().__setattr__(key, value)  # type: ignore
        # Log stuff if desired
        if self._verbose_item_set:
            _log = logger.opt(depth=self._logging_depth)
            _log.trace(f"Set key `{key}` to `{value}`")
            assert getattr(self, key) == value
        # When this method is used as patch, it will halt here
        if self._avoid_recursive:
            return
        # Also update Streamlit's state
        self.session_state[key] = value

    def __setitem__(self, key: str, value: ty.Any) -> None:
        """Set the value of the given key."""
        self.__set__(key, value)

    def __setattr__(self, key: str, value: ty.Any) -> None:
        """Set the value of the given key."""
        self.__set__(key, value)

    def to_dict(self) -> dict[str, ty.Any]:
        """Dump model."""
        return self.model_dump()


class SessionState(metaclass=Singleton):
    """Custom Streamlit session state, in order to revise all present keys, validate them, etc.

    Instances of this class are fully synced with Streamlit's session state. This means that changes to `import streamlit as st; st.session_state` will be reflected in instances of this class, and reverse.

    This class is a singleton, so only one instance can be created at a time.

    Technically, this is a wrapper around the `pydantic` model :class:`_SessionState`.
    """

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton."""
        Singleton.reset(cls)

    def __init__(
        self,
        reverse: bool = DEFAULT_REVERSE_SYNC,
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
        assert isinstance(state, _SessionState)
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

    def get(self, key: str) -> ty.Any:
        """Getter."""
        return self.state[key]

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

    def __enter__(self) -> "SessionState":
        """Activate state temporarily."""
        return self

    def __exit__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        """Reset state."""
        type(self).reset()
