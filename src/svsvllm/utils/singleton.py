__all__ = ["Singleton"]

import typing as ty
from loguru import logger
import threading


class Singleton:
    """This class is implementing the Singleton design pattern in Python.
    The Singleton pattern ensures that a class has only one instance and provides a global point of access to that instance.

    In summary, this class ensures that for any class that uses it as its metaclass, only one instance of that class will ever exist, and subsequent calls to create instances of that class will return the same instance.
    """

    _instance = None
    _lock = threading.Lock()  # A lock to ensure thread safety

    def __new__(cls, *args: ty.Any, **kwargs: ty.Any) -> ty.Any:
        """The method is invoked when an instance of a class with `Singleton` metaclass is created.
        It checks whether the class (`cls`) already exists in `_instances`.
        If the class does not exist, it creates a new instance of the class and adds it to the `_instances` dictionary with the class type as the key.
        Finally, it returns the instance of the class from `_instances`.
        """
        with cls._lock:
            if cls._instance is None:
                logger.trace(f"Creating new {cls}")
                cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        logger.trace(f"Returning {cls._instance}")
        return cls._instance

    # def __call__(cls, *args: ty.Any, **kwargs: ty.Any) -> ty.Any:
    #     """The method is invoked when an instance of a class with `Singleton` metaclass is created.
    #     It checks whether the class (`cls`) already exists in `_instances`.
    #     If the class does not exist, it creates a new instance of the class and adds it to the `_instances` dictionary with the class type as the key.
    #     Finally, it returns the instance of the class from `_instances`.
    #     """
    #     with cls._lock:
    #         if cls._instance is None:
    #             logger.trace(f"Creating new {cls}")
    #             cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
    #     logger.trace(f"Returning {cls._instance}")
    #     return cls._instance
