from unittest import mock

import pytest
from loguru import logger
from streamlit.testing.v1 import AppTest

import svsv


@pytest.mark.parametrize("prompt", ["Hi!", "foo", "bar"])
@pytest.mark.parametrize("default_response", ["Bye!", "pyu", "fyu"])
def test_ui_chat_from_app(app: AppTest, prompt: str, default_response: str) -> None:
    """Run UI function via app and check no exceptions are raised."""
    # Run app
    logger.info(f"Testing {svsv.run_ui}: {app._script_path or app._function}")
    app.run(timeout=15)

    # Patch default LLM response
    with mock.patch.object(svsv.settings, "default_response", default_response):
        # Set value for `chat_input` and run again
        app.chat_input[0].set_value(prompt).run(timeout=10)

    # Log
    logger.info(svsv.session_state)
    logger.info(svsv.session_state.session_state)
    logger.info(svsv.session_state.messages)

    # Test user prompt is there
    assert svsv.session_state.messages
    assert svsv.session_state.messages[-2].role == "user"
    assert svsv.session_state.messages[-2].content == prompt

    # Test user prompt is there
    assert svsv.session_state.messages
    assert svsv.session_state.messages[-1].role == "assistant"
    assert svsv.session_state.messages[-1].content == default_response


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
