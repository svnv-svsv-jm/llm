from __future__ import annotations

__all__ = ["Singleton"]

import typing as ty
from loguru import logger
import threading

_T = ty.TypeVar("_T", bound="Singleton")


class Singleton(type, ty.Generic[_T]):
    """This class is implementing the Singleton design pattern in Python.
    The Singleton pattern ensures that a class has only one instance and provides a global point of access to that instance.

    In summary, this class ensures that for any class that uses it as its metaclass, only one instance of that class will ever exist, and subsequent calls to create instances of that class will return the same instance.
    """

    _instances: dict[type[_T], _T] = {}
    _lock = threading.Lock()  # A lock to ensure thread safety

    def __call__(cls, *args: ty.Any, **kwargs: ty.Any) -> _T:
        """This method is called on object creation."""
        # Lock creation
        with cls._lock:
            # Check if an instance already exists
            if cls._instances.get(cls, None) is None:
                logger.trace(f"Creating instance of {cls}")
                # If not, create a new instance by calling the superclass's __call__
                instance = super(Singleton, cls).__call__(*args, **kwargs)
                cls._instances[cls] = instance
            else:
                logger.trace(f"Instance of {cls} already exists: {cls._instances[cls]}")
        return cls._instances[cls]

    @classmethod
    def reset(cls, key: type) -> None:
        """Reset the singleton so that a new one can be created."""
        logger.debug(f"Resetting: {key}")
        cls._instances[key] = None
        logger.debug(f"New instance registry: {cls._instances}")
