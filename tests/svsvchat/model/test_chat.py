import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os

from svsvchat.session_state import SessionState
from svsvchat.model import create_chat_model
from svsvllm.types import ChatModelType


def test_create_chat_model_cuda(session_state: SessionState, tiny_llama_model_id: str) -> None:
    """Test `create_chat_model` with CUDA-like settings."""
    with patch.object(session_state, "model_name", tiny_llama_model_id):
        chat_model = create_chat_model(use_mlx=False)
    assert isinstance(chat_model, ChatModelType)


def test_create_chat_model_mps(session_state: SessionState, mlx_model_id: str) -> None:
    """Test `create_chat_model` with CUDA-like settings."""
    with patch.object(session_state, "model_name", mlx_model_id):
        chat_model = create_chat_model(use_mlx=True)
    assert isinstance(chat_model, ChatModelType)


def test_create_chat_model(session_state: SessionState, model_id: str, use_mlx: bool) -> None:
    """Test `create_chat_model`."""
    with patch.object(session_state, "model_name", model_id):
        chat_model = create_chat_model(use_mlx=use_mlx)
    assert isinstance(chat_model, ChatModelType)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
