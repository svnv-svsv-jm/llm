from unittest import mock

import pytest
from loguru import logger
from streamlit.testing.v1 import AppTest

import svsv


@pytest.mark.parametrize("prompt", ["Hi!"])
@pytest.mark.parametrize("stream_chat", [["yo", "abc ", "def"]])
def test_ui_chat_from_app(
    app: AppTest,
    prompt: str,
    stream_chat: list[str],
    patch_ollama: mock.MagicMock,
) -> None:
    """Run UI function via app and check no exceptions are raised."""
    # Run app
    logger.info(f"Testing {svsv.run_ui}: {app._script_path or app._function}")

    # Patch default LLM response
    # Run app once or `chat_input` element will not be there
    app.run(timeout=15)
    # Set value for `chat_input` and run again
    app.chat_input[0].set_value(prompt).run(timeout=10)

    # Log
    logger.info(f"Session state: {svsv.session_state}")
    logger.info(f"Streamlit session state: {svsv.session_state.session_state}")
    logger.info(f"Chat history: {svsv.session_state.messages}")
    logger.info(f"Stream chat: {stream_chat}")

    # Test patch applied correctly
    stream_chat_mock = patch_ollama.stream_chat
    assert isinstance(stream_chat_mock, mock.MagicMock)
    stream_chat_mock.assert_called()

    # Test no exceptions
    logger.info(f"Exc: {app.exception}")
    assert not app.exception

    # Test there is a markdown component
    assert app.markdown[0].value == prompt

    # Test user prompt is there
    assert svsv.session_state.messages
    assert svsv.session_state.messages[-2].role == "user"
    assert svsv.session_state.messages[-2].content == prompt

    # Test LLM response
    assert svsv.session_state.llm is not None
    assert svsv.session_state.messages
    assert svsv.session_state.messages[-1].role == "assistant"
    assert svsv.session_state.messages[-1].content == "".join(stream_chat)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
