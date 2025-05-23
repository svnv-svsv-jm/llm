__all__ = ["TokenizerType", "ModelType", "UuidType", "StateType", "Languages", "ChatModelType"]

import typing as ty
import uuid

import torch
from llama_index.core.llms.custom import CustomLLM
from pydantic import AfterValidator
from streamlit.runtime.state import (
    SafeSessionState,
    SessionStateProxy,
)
from streamlit.runtime.state import (
    SessionState as StreamlitSessionState,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, SpecialTokensMixin

# mlx is optional
try:  # pragma: no cover
    from mlx.nn import Module  # type: ignore
    from mlx_lm.tokenizer_utils import TokenizerWrapper
except ImportError:  # pragma: no cover
    TokenizerWrapper = AutoTokenizer
    Module = torch.nn.Module

TokenizerType = AutoTokenizer | SpecialTokensMixin | torch.nn.Module | Module | TokenizerWrapper
ModelType = AutoModelForCausalLM | torch.nn.Module | Module
UuidType = ty.Annotated[str, AfterValidator(lambda x: str(uuid.UUID(x)))]
StateType = StreamlitSessionState | SessionStateProxy | SafeSessionState
Languages = ty.Literal["English", "Italian"]
ChatModelType = CustomLLM
