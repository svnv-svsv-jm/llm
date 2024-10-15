__all__ = [
    "patch_torch_quantized_engine",
    "mock_hf_model_creation",
    "mock_transformers_pipeline",
    "mock_hf_pipeline",
    "mock_hf_chat",
]

import pytest
from unittest.mock import patch, MagicMock
import typing as ty
from importlib import reload

import torch
from torch.backends import quantized
import huggingface_hub  # login
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from optimum.quanto import QuantizedModelForCausalLM
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


@pytest.fixture
def patch_torch_quantized_engine(device: torch.device) -> ty.Generator[bool, None, None]:
    """Workaround to try quantization on Mac M1."""
    if device == torch.device("mps"):
        try:
            with patch.object(quantized, "engine", new_value="qnnpack"):
                assert quantized.engine == "qnnpack"
                yield True
        except:
            yield True
    else:
        yield False


@pytest.fixture
def mock_hf_model_creation() -> ty.Iterator[dict[str, MagicMock]]:
    """Mock creation of a model from HuggingFace."""
    with patch.object(
        huggingface_hub,
        "login",
    ) as mock_login, patch.object(
        _BaseAutoModelClass,
        "from_pretrained",
        return_value=MagicMock(),
    ) as mock_model_from_pretrained, patch.object(
        AutoModelForCausalLM,
        "from_pretrained",
        return_value=MagicMock(),
    ) as mock_automodel_from_pretrained, patch.object(
        AutoTokenizer,
        "from_pretrained",
        return_value=MagicMock(),
    ) as mock_tokenizer_from_pretrained, patch.object(
        QuantizedModelForCausalLM,
        "from_pretrained",
        return_value=MagicMock(),
    ) as mock_qmodel_from_pretrained, patch.object(
        QuantizedModelForCausalLM,
        "quantize",
        return_value=MagicMock(),
    ) as mock_qmodel_quantize, patch(
        "torch.quantization.quantize_dynamic"
    ) as mock_quantize_dynamic:
        yield {
            "mock_login": mock_login,
            "mock_automodel_from_pretrained": mock_automodel_from_pretrained,
            "mock_model_from_pretrained": mock_model_from_pretrained,
            "mock_tokenizer_from_pretrained": mock_tokenizer_from_pretrained,
            "mock_qmodel_from_pretrained": mock_qmodel_from_pretrained,
            "mock_qmodel_quantize": mock_qmodel_quantize,
            "mock_quantize_dynamic": mock_quantize_dynamic,
        }


@pytest.fixture
def mock_transformers_pipeline() -> ty.Iterator[MagicMock]:
    """Mock `transformers.pipeline`."""
    reload(transformers)  # Reload the submodule to apply the patch globally
    with patch.object(
        transformers,
        "pipeline",
    ) as mock:
        yield mock


@pytest.fixture
def mock_hf_pipeline() -> ty.Iterator[MagicMock]:
    """Mock `langchain_huggingface.HuggingFacePipeline`."""
    with patch.object(HuggingFacePipeline, "__new__", return_value=MagicMock()) as mock:
        yield mock


@pytest.fixture
def mock_hf_chat() -> ty.Iterator[MagicMock]:
    """Mock `langchain_huggingface.ChatHuggingFace`."""
    with patch.object(ChatHuggingFace, "__new__", return_value=MagicMock()) as mock:
        yield mock
