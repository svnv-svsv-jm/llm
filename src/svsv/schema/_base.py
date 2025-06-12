__all__ = ["BaseModelWithValidation"]

import pydantic
from loguru import logger
from pydantic import BaseModel


class BaseModelWithValidation(BaseModel):
    """Just adds a `is_valid` method."""

    @classmethod
    def is_valid(cls, obj: dict) -> bool:
        """`True` if input complies to schema."""
        with logger.catch(pydantic.ValidationError):
            cls.model_validate(obj)
            return True
        return False
