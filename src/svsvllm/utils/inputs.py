__all__ = ["get_fn_default_value", "pop_params_not_in_fn"]

import typing as ty
from inspect import signature
from loguru import logger
import pprint


def get_fn_default_value(fn: ty.Callable, name: str) -> ty.Any:
    """Gets default value for a given function.

    Args:
        fn (ty.Callable):
            Input function

        name (str):
            Name of the function's argument we want the default value for.

    Raises:
        ValueError: If the function has not argument named `name`.

    Returns:
        ty.Any: Value of the function's default value.
    """
    # Get the signature of the function
    signature_ = signature(fn)

    # Retrieve default value of specified parameter
    for param in signature_.parameters.values():
        if param.default is not param.empty and param.name == name:
            return param.default

    raise ValueError(f"Could not find parameter with name '{name}'")


def pop_params_not_in_fn(fn: ty.Callable, params: dict) -> dict:
    """Pops parameters not in callable's signature.

    Args:
        fn (Callable):
            Input callable, of which we get the signature.

        params (dict):
            Keyword arguments that have to be in the callable's signature, or they'll be popped.

    Returns:
        (dict): Sanitized dictionary containing only the keys and values of `params` that are present in the `fn`'s signature.
    """
    sig = signature(fn)
    logger.trace(f"Signature: {sig}")
    inputs = params.copy()
    for name, _ in params.items():
        if name not in sig.parameters:
            logger.trace(f"Popping {name}")
            inputs.pop(name, None)
    logger.trace(f"Returning: {pprint.pformat(inputs, indent=2)}")
    return inputs
