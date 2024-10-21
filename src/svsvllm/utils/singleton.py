__all__ = ["Singleton"]

import typing as ty
from loguru import logger
import threading


class Singleton(type):
    """This class is implementing the Singleton design pattern in Python.
    The Singleton pattern ensures that a class has only one instance and provides a global point of access to that instance.

    In summary, this class ensures that for any class that uses it as its metaclass, only one instance of that class will ever exist, and subsequent calls to create instances of that class will return the same instance.
    """

    _instances: dict[ty.Type, ty.Any] = {}
    _lock = threading.Lock()  # A lock to ensure thread safety

    def __call__(cls, *args: ty.Any, **kwargs: ty.Any) -> ty.Any:
        """This method is called on object creation."""
        logger.debug(f"Creating {cls} with {kwargs}")
        # Lock creation
        with cls._lock:
            # Check if an instance already exists
            if cls._instances.get(cls, None) is None:
                # If not, create a new instance by calling the superclass's __call__
                instance = super(Singleton, cls).__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton."""
        cls._instances[cls] = None
