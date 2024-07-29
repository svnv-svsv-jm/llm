import pytest
from loguru import logger
import sys, os
import typing as ty

import time
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder

from svsvllm.scraping import (
    DRIVER_TYPE,
    write_keyword_in_search_bar,
    click_on_search,
    download_pdfs_on_page,
)
from svsvllm.scraping.url import AE_URL


def test_ae_website_search(
    web_driver: DRIVER_TYPE,
    artifact_location: str,
) -> None:
    """Test the driver."""
    # Go to desired URL
    web_driver.get(AE_URL)
    time.sleep(5)

    # Get page's HTML code for debugging
    html = BeautifulSoup(web_driver.page_source, "html.parser").prettify()
    logger.info(f"Page source:\n{html}")

    # Test search bar
    key = "banca"
    value = write_keyword_in_search_bar(key, web_driver=web_driver)
    assert value == key

    # Click on search
    button = click_on_search(web_driver)
    assert button is not None

    # Download files
    has_clicked = download_pdfs_on_page(web_driver)
    assert has_clicked


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
