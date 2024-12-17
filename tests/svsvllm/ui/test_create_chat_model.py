import pytest
from unittest.mock import patch
from loguru import logger
import typing as ty
import sys, os

from streamlit.testing.v1 import AppTest
from langchain_core.language_models.chat_models import BaseChatModel
import torch
from transformers import BitsAndBytesConfig


from svsvllm.ui.model import create_chat_model
from svsvllm.defaults import DEFAULT_LLM, ZEPHYR_CHAT_TEMPLATE


@pytest.mark.parametrize("apply_chat_template", [False, True])
@pytest.mark.parametrize("chat_template", [ZEPHYR_CHAT_TEMPLATE])
@pytest.mark.parametrize("use_mlx", [True, False])
def test_create_chat_model(
    apptest_ss: AppTest,  # Import just to bind session state
    bnb_config: BitsAndBytesConfig,
    device: torch.device,
    pipeline_kwargs: dict,
    apply_chat_template: bool,
    chat_template: str | dict[str, ty.Any] | list[dict[str, ty.Any]],
    use_mlx: bool,
    model_id: str,
    mlx_model_id: str,
) -> None:
    """Test we can query the created LLM."""
    # Create chat model
    chat_model = create_chat_model(
        mlx_model_id if use_mlx else model_id,
        apply_chat_template=apply_chat_template,
        chat_template=chat_template,
        pipeline_kwargs=pipeline_kwargs,
        device=device,
        bnb_config=bnb_config,
        quantize=True,
        quantize_w_torch=True,
        use_mlx=use_mlx,
    )
    logger.info(f"Chat model: {chat_model}")
    assert isinstance(chat_model, BaseChatModel)

    # Get LLM
    llm = getattr(chat_model, "llm", None)
    logger.info(f"LLM: {llm}")

    logger.success("ok")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
