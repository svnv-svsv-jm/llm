__all__ = ["set_up_logging", "Formatter", "serialize", "formatter"]

import sys, os
import typing as ty
from loguru import logger
from loguru._defaults import LOGURU_FORMAT
import logging
import json

DEFAULT_FORMAT = f"{LOGURU_FORMAT}" + " | <yellow>{extra}</yellow>\n"
DEFAULT_LOG_LEVEL = "INFO"


def set_up_logging(
    level: str | None = None,
    serialize: bool = False,
    show_file_info: bool = False,
    remove: bool = True,
) -> ty.Any:
    """Set up logging."""
    # Remove
    if remove:
        logger.remove()  # pragma: no cover

    # Get logging level
    if level is None:
        level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)

    # Set up logger
    logger_id = logger.add(
        sys.stdout,
        level=level.upper(),
        format=Formatter(serialize=serialize, show_file_info=show_file_info),  # type: ignore
    )
    return logger_id


def serialize(record: logging.LogRecord, show_file_info: bool = False) -> str:
    """Create subset."""
    # Extract needed info
    file_path = record["file"].path  # type: ignore
    name = record["name"]  # type: ignore
    module = record["module"]  # type: ignore
    function = record["function"]  # type: ignore
    line = record["line"]  # type: ignore
    if show_file_info:
        line_info = f"{file_path}:{line}"
    else:
        line_info = f"{name}.{module}.{function}:{line}"
    # Create set
    subset = {
        "timestamp": record["time"].timestamp(),  # type: ignore
        "message": f'{line_info} | {record["message"]}',  # type: ignore
        "log.level": record["level"].name,  # type: ignore
        "file": file_path,
        "name": name,
        "module": module,
        "function": function,
        "line": line,
        "ecs": {"version": "1.6.0"},
        "extra": record["extra"],  # type: ignore
    }
    return json.dumps(subset)


def formatter(record: logging.LogRecord) -> str:
    """Note this function returns the string to be formatted, not the actual message to be logged."""
    record["extra"]["serialized"] = serialize(record)  # type: ignore
    return "{extra[serialized]}\n"


class Formatter:
    """Custom formatter for `loguru`, to get the right serialized logs for ElasticSearch."""

    def __init__(self, serialize: bool = False, show_file_info: bool = False) -> None:
        """
        Args:
            serialize (bool):
                Whether to serialize logs for ElasticSearch or not.

            show_file_info (bool, optional):
                Whether to add information about the file path where the log statement is coming from.
                If `False`, only the module and function and the line will be shown, not the full path.
                Defaults to `False`.
        """
        self.serialize = serialize
        self.show_file_info = show_file_info

    def __call__(self, record: logging.LogRecord) -> str:
        """Main."""
        if self.serialize:
            return formatter(record)
        return DEFAULT_FORMAT
