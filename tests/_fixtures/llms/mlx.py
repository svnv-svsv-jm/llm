__all__ = ["mlx_model_id", "mlx_llm"]

import pytest
import typing as ty
from loguru import logger

from langchain_community.llms.mlx_pipeline import MLXPipeline


@pytest.fixture
def mlx_model_id() -> str:
    """MLX model ID."""
    return "mlx-community/quantized-gemma-2b-it"


@pytest.fixture
def mlx_llm(mlx_model_id: str) -> MLXPipeline:
    """MLX LLM."""
    llm = MLXPipeline.from_model_id(
        mlx_model_id,
        pipeline_kwargs={
            "temp": 0.1,
            "max_new_tokens": 2**10,
        },
    )
    return llm
