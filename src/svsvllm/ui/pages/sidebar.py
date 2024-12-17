__all__ = ["sidebar"]

import os
import typing as ty
from loguru import logger
import streamlit as st

from svsvllm.defaults import DEFAULT_LLM, OPENAI_DEFAULT_MODEL, EMBEDDING_DEFAULT_MODEL
from svsvllm.const import PageNames
from ..file_upload import file_uploader
from ..locale import LANGUAGES
from ..callbacks import PageSelectorCallback, UpdateLanguageCallback
from ..session_state import session_state


def sidebar() -> None:
    """Sets up the sidebar."""
    with st.sidebar:
        # Information
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        # st.markdown(
        #     "[View the source code](https://github.com/svnv-svsv-jm/llm/blob/main/src/svsvllm/app/ui/ui.py)"
        # )
        st.markdown(
            "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/svnv-svsv-jm/llm?quickstart=1)"
        )

        # Create a selectbox for language selection
        logger.trace(f"Language selection: {session_state.language}")
        st.selectbox(
            label="Select a Language",
            options=LANGUAGES,
            key="new_language",
            index=LANGUAGES.index(session_state.language),
            on_change=UpdateLanguageCallback("language-update"),
        )

        # OpenAI key
        st.text_input("OpenAI API Key", key="openai_api_key", type="password")

        # Model name
        st.text_input(
            "OpenAI LLM name",
            key="openai_model_name",
            placeholder=OPENAI_DEFAULT_MODEL,
        )
        st.text_input(
            "HuggingFace LLM name",
            key="model_name",
            placeholder=session_state.model_name,
        )
        st.text_input(
            "RAG embedding model name",
            key="embedding_model_name",
            placeholder=session_state.embedding_model_name,
        )

        # File uploader for multiple files (simulate folder upload)
        file_uploader()

        # Button to go to the settings page
        st.button(
            "Go to Settings",
            on_click=PageSelectorCallback(PageNames.SETTINGS, name="page-selector"),
        )
