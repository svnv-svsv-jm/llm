import typing as ty
import torch
from transformers import AutoTokenizer, SpecialTokensMixin, AutoModelForCausalLM
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx.nn import Module  # type: ignore
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import AfterValidator
import uuid

TokenizerType = AutoTokenizer | SpecialTokensMixin | torch.nn.Module | Module | TokenizerWrapper
ModelType = AutoModelForCausalLM | torch.nn.Module | Module
ChatModelType = BaseChatModel
UuidType = ty.Annotated[str, AfterValidator(lambda x: str(uuid.UUID(x)))]
