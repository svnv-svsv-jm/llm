__all__ = ["BaseModelWithValidation"]

import typing as ty
from pydantic import BaseModel


class BaseModelWithValidation(BaseModel):
    """Just adds a `is_valid` method."""

    @classmethod
    def is_valid(cls, obj: dict) -> bool:
        """`True` if input complies to schema."""
        try:
            cls.model_validate(obj)
            return True
        except:
            return False
