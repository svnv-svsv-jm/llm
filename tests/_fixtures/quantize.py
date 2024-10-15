__all__ = ["bnb_config"]

import pytest
from loguru import logger

import torch
from transformers import BitsAndBytesConfig


@pytest.fixture
def bnb_config() -> BitsAndBytesConfig | None:
    """Quantization configuration with `BitsAndBytes`."""
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        bnb_config = None
    logger.debug(f"Config: {bnb_config}")
    return bnb_config
