import pytest
from loguru import logger
import typing as ty
import sys

from svsvllm.loaders import load_tokenizer
from svsvllm.utils import add_chat_template
from svsvllm.defaults import DEFAULT_LLM, ZEPHYR_CHAT_TEMPLATE


@pytest.mark.parametrize("model_name", [DEFAULT_LLM])
@pytest.mark.parametrize("chat_template", [ZEPHYR_CHAT_TEMPLATE])
def test_add_chat_template(
    model_name: str,
    chat_template: str | dict[str, ty.Any] | list[dict[str, ty.Any]],
) -> None:
    """Test `add_chat_template`."""
    tokenizer = load_tokenizer(model_name)
    tokenizer = add_chat_template(
        tokenizer,
        chat_template=chat_template,
    )
    assert tokenizer.chat_template is not None


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
