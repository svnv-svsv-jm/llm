__all__ = ["add_chat_template"]

import typing as ty
from loguru import logger

from transformers import PreTrainedTokenizerBase


def add_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    chat_template: str | dict[str, ty.Any] | list[dict[str, ty.Any]],
    apply_chat_template: bool = True,
    force_chat_template: bool = False,
) -> PreTrainedTokenizerBase:
    """Apply tokenizer template.

    Args:
        tokenizer (PreTrainedTokenizerBase):
            Tokenizer to add the template to.

        apply_chat_template (bool, optional):
            Whether to apply chat template to tokenizer.
            Defaults to `True`.

        chat_template (str | dict[str, ty.Any] | list[dict[str, ty.Any]]):
            Chat template to enforce when a default one is not available.
            Defaults to `False`.

        force_chat_template (bool):
            If `True`, the provided chat template will be forced on the tokenizer.
            Defaults to `False`.
    """
    template = getattr(tokenizer, "chat_template", None)
    logger.trace(f"Tokenizer chat template: {template}")
    if apply_chat_template:
        if template is None or force_chat_template:
            logger.trace(f"Applying template:\n{chat_template}")
            tokenizer.chat_template = chat_template
        else:
            logger.trace(f"Cannot force chat template, tokenizer already has one: {template}")
    return tokenizer
