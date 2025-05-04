from pprint import pprint

import pytest
from loguru import logger


@pytest.mark.parametrize("log_level", ["TRACE", "DEBUG", "INFO"])
@pytest.mark.parametrize("message", ["foo", "bar"])
def test_logs_can_be_caught(
    log_level: str,
    catch_logs: list[dict],
    message: str,
) -> None:
    """Test we can capture logs, thus test logs exist."""
    logger.log(log_level, message)

    pprint(catch_logs, indent=2)

    assert message == catch_logs[0]["record"]["message"]
    assert log_level == catch_logs[0]["record"]["level"]["name"]


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
