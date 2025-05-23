__all__ = ["configure_logger", "settings", "Settings", "session_state", "run_ui", "set_up_logging"]

import os
from importlib import metadata

from loguru import logger

from ._session_state import session_state
from ._settings import Settings, settings
from .ui import run_ui
from .utils import set_up_logging

package_name = os.path.basename(os.path.dirname(__file__))
logger.disable(package_name)


def configure_logger(enable: bool = False) -> None:
    """Configure logging."""
    logger.disable(package_name)
    if enable:
        logger.enable(package_name)


v = metadata.version(package_name)
__version__ = f"{v}"
