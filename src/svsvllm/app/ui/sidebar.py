__all__ = ["sidebar"]

import os
import typing as ty
from loguru import logger
import streamlit as st

from .file_upload import file_uploader
from .locale import LANGUAGES
from .defaults import OPENAI_DEFAULT_MODEL, EMBEDDING_DEFAULT_MODEL
from .const import PageNames
from .callbacks import PageSelectorCallback, UpdateLanguageCallback


def sidebar() -> None:
    """Sets up the sidebar."""
    with st.sidebar:
        # Information
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        st.markdown(
            "[View the source code](https://github.com/svnv-svsv-jm/llm/blob/main/src/svsvllm/app/ui/ui.py)"
        )
        st.markdown(
            "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/svnv-svsv-jm/llm?quickstart=1)"
        )

        # Create a selectbox for language selection
        if "language" not in st.session_state:
            st.session_state.language = LANGUAGES[0]
            logger.trace(f"Language selection: {st.session_state.language}")
        st.selectbox(
            label="Select a Language",
            options=LANGUAGES,
            key="new_language",
            index=LANGUAGES.index(st.session_state.language),
            on_change=UpdateLanguageCallback("language-update"),
        )

        # OpenAI key
        st.text_input("OpenAI API Key", key="openai_api_key", type="password")

        # Model name
        st.text_input(
            "LLM name",
            key="model_name",
            placeholder=OPENAI_DEFAULT_MODEL,
        )
        st.text_input(
            "Embedding model name",
            key="embedding_model_name",
            placeholder=EMBEDDING_DEFAULT_MODEL,
        )

        # File uploader for multiple files (simulate folder upload)
        file_uploader()

        # Button to go to the settings page
        st.button(
            "Go to Settings",
            on_click=PageSelectorCallback(PageNames.SETTINGS, name="page-selector"),
        )
