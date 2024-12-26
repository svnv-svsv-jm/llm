__all__ = ["use_mlx", "mlx_llm"]

import pytest
import typing as ty
from loguru import logger

from langchain_community.llms.mlx_pipeline import MLXPipeline


@pytest.fixture
def use_mlx() -> bool:
    """Whether to use `mlx` or not."""
    try:
        import mlx_lm

        MLX_INSTALLED = True
    except ImportError:
        MLX_INSTALLED = False

    return MLX_INSTALLED


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
