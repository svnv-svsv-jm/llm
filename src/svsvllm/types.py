__all__ = [
    "StateModifier",
    "TokenizerType",
    "ModelType",
    "ChatModelType",
    "UuidType",
    "StateType",
    "Languages",
]

import typing as ty
import torch
from transformers import AutoTokenizer, SpecialTokensMixin, AutoModelForCausalLM
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import StateModifier
from pydantic import AfterValidator
import uuid
from streamlit.runtime.state import (
    SafeSessionState,
    SessionStateProxy,
    SessionState as StreamlitSessionState,
)

# mlx is optional
try:  # pragma: no cover
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx.nn import Module  # type: ignore
except ImportError:  # pragma: no cover
    TokenizerWrapper = AutoTokenizer
    Module = torch.nn.Module

TokenizerType = AutoTokenizer | SpecialTokensMixin | torch.nn.Module | Module | TokenizerWrapper
ModelType = AutoModelForCausalLM | torch.nn.Module | Module
ChatModelType = BaseChatModel
UuidType = ty.Annotated[str, AfterValidator(lambda x: str(uuid.UUID(x)))]
StateType = StreamlitSessionState | SessionStateProxy | SafeSessionState
Languages = ty.Literal["English", "Italian"]
