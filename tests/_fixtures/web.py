__all__ = ["web_driver"]

import pytest
import typing as ty
from loguru import logger
import time

from svsvllm.scraping.driver import create_driver, DRIVER_TYPE


@pytest.fixture
def web_driver() -> ty.Iterator[DRIVER_TYPE]:
    """Web driver."""
    # Headless only if running all tests
    headless = True  # __file__ not in sys.argv[0]
    # Create driver
    driver = create_driver(
        "window-size=1200x600",
        headless=headless,
        use_firefox=False,
    )
    logger.debug(f"Created driver: {driver}")
    yield driver
    # Sleep
    time.sleep(1)
    driver.quit()
