import pytest
from unittest.mock import patch, MagicMock
from loguru import logger
import typing as ty
import sys, os
from streamlit.testing.v1 import AppTest

from svsvchat.session_state import SessionState
from svsvchat.settings import settings


@pytest.mark.parametrize("has_chat", [False, True])
@pytest.mark.parametrize("language", ["English", "Italian"])
@pytest.mark.parametrize("mock_chat_input", [["Hi", "Ok", None], [None]], indirect=True)
def test_chat_page(
    session_state: SessionState,
    apptest: AppTest,
    mock_chat_input: MagicMock,
    res_docs_path: str,
    has_chat: bool,
    language: str,
) -> None:
    """Test `main` page setup: title, headers, etc."""
    with patch.object(
        settings,
        "has_chat",
        has_chat,
    ), patch.object(
        settings,
        "uploaded_files_dir",
        res_docs_path,
    ), patch.object(
        session_state,
        "page",
        "main",
    ), patch.object(
        session_state,
        "language",
        language,
    ):
        apptest.run(timeout=30)

    # Test basics
    if apptest.exception:
        ex = apptest.exception[0]
        ex_msg = "\n".join(ex.stack_trace + [ex.message])
        raise Exception(f"{ex_msg}")

    # Test init message
    logger.info(f"Chat history: {session_state.chat_history}")
    init_msg = session_state.chat_history[0].content
    if language.lower() == "italian":
        assert init_msg == settings.start_message_it
    else:
        assert init_msg == settings.start_message_en

    # Test title
    assert len(apptest.title) > 0, "No title"
    title = apptest.title[0]
    logger.info(f"Title: {title.value}")
    assert title.value == settings.app_title

    # Test subheader
    assert len(apptest.subheader) > 0, "No subheader"
    subheader = apptest.subheader[0]
    logger.info(f"Subheader: {subheader.value}")
    assert subheader.value == settings.app_subheader

    # Test subheader
    assert len(apptest.caption) > 0, "No caption"
    caption = apptest.caption[0]
    logger.info(f"Caption: {caption.value}")
    assert caption.value == settings.app_caption

    # Test chat
    if not has_chat:
        return
    mock_chat_input.assert_called()
    assert len(apptest.chat_message) > 0, "No chat message."
    assert len(apptest.markdown) > 0, "No markdown."

    # Success
    logger.success("Ok.")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
