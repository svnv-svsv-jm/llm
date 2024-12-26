__all__ = ["mlx_model_id", "tiny_llama_model_id", "model_id"]

import pytest
from unittest.mock import patch
import typing as ty

from svsvllm.const import DEFAULT_LLM, DEFAULT_LLM_MLX


@pytest.fixture
def mlx_model_id() -> str:
    """MLX model ID."""
    return f"{DEFAULT_LLM_MLX}"


@pytest.fixture
def tiny_llama_model_id() -> str:
    """TinyLlama model ID."""
    return f"{DEFAULT_LLM}"


@pytest.fixture
def model_id(use_mlx: bool, mlx_model_id: str, tiny_llama_model_id: str) -> str:
    """LLM model ID."""
    if use_mlx:
        return mlx_model_id
    return tiny_llama_model_id
