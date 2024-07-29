import pytest
import sys
import typing as ty
from loguru import logger
import time

from svsvllm.scraping.driver import create_driver, DRIVER_TYPE


@pytest.fixture
def artifact_location() -> str:
    """Location for test artifacts."""
    return "pytest_artifacts"


@pytest.fixture
def web_driver() -> ty.Generator[DRIVER_TYPE, None, None]:
    """Web driver."""
    # Headless only if running all tests
    headless = False  # __file__ not in sys.argv[0]
    # Create driver
    driver = create_driver(
        "window-size=1200x600",
        headless=headless,
        use_firefox=False,
    )
    yield driver
    # Sleep
    time.sleep(2)
    driver.quit()
