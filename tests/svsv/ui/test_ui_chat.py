import pytest
from loguru import logger
from streamlit.testing.v1 import AppTest

import svsv


@pytest.mark.parametrize("prompt", ["Hi!", "foo", "bar"])
def test_ui_basics(app: AppTest, prompt: str) -> None:
    """Run UI function via app and check no exceptions are raised."""
    # Run app
    logger.info(f"Testing {svsv.run_ui}: {app._script_path or app._function}")
    app.run(timeout=15)

    # Test there is a `chat_input` component
    assert app.chat_input, "No chat_input component found"

    # Set value for `chat_input` and run again
    app.chat_input[0].set_value(prompt).run(timeout=10)

    # Test there is a markdown component
    assert app.markdown[0].value == prompt

    # Test the LLM responded
    logger.info(svsv.session_state)
    logger.info(svsv.session_state.session_state)
    assert svsv.session_state.messages
    assert svsv.session_state.messages[-1].role == "user"


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
