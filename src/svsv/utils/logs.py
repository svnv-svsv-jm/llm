__all__ = ["set_up_logging"]

import sys

from loguru import logger

import svsv


def set_up_logging(log_level: int | str | None = None) -> int:
    """Sets up logging.

    Args:
        log_level (int | str, optional):
            Logging level.
            Defaults to `svsv.settings.log_level`.

    Returns:
        int: Logger ID.
    """
    log_level = svsv.settings.log_level if log_level is None else log_level
    logger.remove()
    return logger.add(sys.stderr, level=svsv.settings.log_level)
