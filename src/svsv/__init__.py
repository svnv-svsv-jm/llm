__all__ = ["configure_logger"]

import os
from loguru import logger
from importlib import metadata

package_name = os.path.basename(os.path.dirname(__file__))
logger.disable(package_name)


def configure_logger(enable: bool = False) -> None:
    """Configure logging."""
    logger.disable(package_name)
    if enable:
        logger.enable(package_name)


v = metadata.version(package_name)
__version__ = f"{v}"
