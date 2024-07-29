__all__ = ["click_on_search"]

from loguru import logger
import typing as ty
import time

from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .driver import DRIVER_TYPE


def click_on_search(
    web_driver: DRIVER_TYPE,
    button_text: str = "Ricerca",
    button_class: str = "btn-primary",
    timeout: float = 10,
    sleep: float = 1,
) -> WebElement:
    """Clicks on the search button.

    Args:
        web_driver (DRIVER_TYPE):
            Current web driver.

        button_text (str):
            Default text written in the button.
            Used to identify the button.
            Defaults to `"Ricerca"`.

        button_class (str):
            Class of the button.
            Used to identify the button.
            Defaults to `"btn-primary"`.

        timeout (float, optional):
            Timeout value for button to become cliackable.
            Defaults to `10`.

        sleep (float, optional):
            Sleeping time in seconds after clicking.
            Defaults to `1`.

    Returns:
        WebElement:
            The retrieved button.
    """
    # Find button
    XPATH = f"//button[@type='button' and contains(@class, '{button_class}') and text()='{button_text}']"
    button = web_driver.find_element(By.XPATH, XPATH)
    logger.info(f"Element: {button}")
    # Scroll the element into view
    web_driver.execute_script("arguments[0].scrollIntoView();", button)
    # Wait until the button is clickable
    wait = WebDriverWait(web_driver, timeout)
    button = wait.until(EC.element_to_be_clickable(button))
    # Click
    web_driver.execute_script("arguments[0].click();", button)
    time.sleep(sleep)
    return button
