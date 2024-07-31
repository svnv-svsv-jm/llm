__all__ = ["DRIVER_TYPE", "create_driver"]

import typing as ty
from loguru import logger

from selenium.webdriver import Chrome, ChromeOptions, FirefoxOptions, Firefox
from selenium.webdriver import Chrome, Firefox

DRIVER_TYPE = Firefox | Chrome


def create_driver(
    *args: ty.Any,
    disable_notifications: bool = True,
    headless: bool = True,
    use_firefox: bool = False,
) -> DRIVER_TYPE:
    """Create the web driver.

    Args:
        disable_notifications (bool, optional):
            Defaults to `True`.

        headless (bool, optional):
            Whether a popup window for the chosen browser will pop up.
            Defaults to `True`.

        use_firefox (bool, optional):
            Whether to use Firefox or Chrome.
            Defaults to `False`.

    Raises:
        TypeError: If unknown options are found.

    Returns:
        Firefox | Chrome: The web driver.
    """
    logger.trace("Creating web driver")
    options = ChromeOptions() if not use_firefox else FirefoxOptions()

    # Parse arguments
    for arg in args:
        logger.trace(f"Adding argument {arg}")
        options.add_argument(arg)
    if disable_notifications:
        logger.trace("disable-notifications")
        options.add_argument("disable-notifications")
    if headless:
        logger.trace("--headless")
        options.add_argument("--headless")

    # Create driver
    if isinstance(options, ChromeOptions):
        logger.trace(f"Creating driver {Chrome}")
        return Chrome(options=options)

    if isinstance(options, FirefoxOptions):
        logger.trace(f"Creating driver {Firefox}")
        return Firefox(options=options)

    # Error if not supported
    raise TypeError(f"Unknown options of type {type(options)}")
