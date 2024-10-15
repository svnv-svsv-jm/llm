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

from svsvllm.scraping.driver import DRIVER_TYPE
from svsvllm.scraping.url import AE_URL


@pytest.mark.integtest
def test_now_secure(
    web_driver: DRIVER_TYPE,
    artifact_location: str,
) -> None:
    """Test now secure."""
    # Test now seecure
    web_driver.get("https://nowsecure.nl")
    web_driver.save_screenshot(os.path.join(artifact_location, "nowsecure.png"))


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
