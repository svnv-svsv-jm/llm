import pytest
from loguru import logger
from streamlit.testing.v1 import AppTest

import svsv


def test_ui_basics(app: AppTest) -> None:
    """Run UI function via app and check no exceptions are raised."""
    logger.info(f"Testing {svsv.run_ui}: {app._script_path or app._function}")
    app.run(timeout=15)
    assert app.chat_input, "No chat_input component found"
    app.chat_input[0].set_value("Hi").run(timeout=10)
    assert not app.exception


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
