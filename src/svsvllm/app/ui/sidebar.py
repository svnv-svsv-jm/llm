__all__ = ["sidebar"]

import typing as ty
import streamlit as st

from .locale import LANGUAGES
from .defaults import DEFAULT_MODEL
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
        st.selectbox(
            label="Select a Language",
            options=LANGUAGES,
            key="new_language",
            index=LANGUAGES.index(st.session_state.language),
            on_change=UpdateLanguageCallback(),
        )

        # OpenAI key
        st.text_input("OpenAI API Key", key="openai_api_key", type="password")

        # Model name
        st.text_input(
            "LLM name",
            key="model_name",
            placeholder=DEFAULT_MODEL,
        )

        # File uploader for multiple files (simulate folder upload)
        st.file_uploader(
            "Upload files from a folder",
            accept_multiple_files=True,
            key="uploaded_files",
        )

        # Button to go to the settings page
        st.button("Go to Settings", on_click=PageSelectorCallback(PageNames.SETTINGS))
