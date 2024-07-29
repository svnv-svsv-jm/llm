__all__ = ["write_keyword_in_search_bar"]

from loguru import logger
import typing as ty
import time

from selenium.webdriver.common.by import By

from .driver import DRIVER_TYPE


def write_keyword_in_search_bar(key: str, web_driver: DRIVER_TYPE, sleep: float = 1) -> str | None:
    """Writes keyword in search bar.

    Args:
        key (str):
            Keyword in search bar to send.

        web_driver (DRIVER_TYPE):
            Current web driver.

        sleep (float, optional):
            Sleeping time in seconds after sending the key to the driver. Defaults to `1`.

    Returns:
        str | None:
            The retrieved value, supposed to be equal to the sent one.
    """
    # Find the bar where we can write what to search for
    elem = web_driver.find_element(by=By.ID, value="input-group-dropdown-1")
    logger.debug(f"Element: {elem}")
    # Write search keyword
    elem.send_keys(key)
    time.sleep(sleep)
    # Test keyword was written successfully
    value = elem.get_attribute("value")
    logger.debug(f"Element:\n\tattribute: {value}")
    return value
