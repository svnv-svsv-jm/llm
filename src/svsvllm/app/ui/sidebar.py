__all__ = ["sidebar"]

import typing as ty
import streamlit as st

from .locale import LANGUAGES


def sidebar() -> dict[str, ty.Any]:
    """Sets up the sidebar."""
    with st.sidebar:
        # Create a selectbox for language selection
        selected_language = st.selectbox("Select a Language", LANGUAGES)

        # OpenAI key
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

        # File uploader for multiple files (simulate folder upload)
        uploaded_files = st.file_uploader("Upload files from a folder", accept_multiple_files=True)

        # Information
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        st.markdown(
            "[View the source code](https://github.com/svnv-svsv-jm/llm/blob/main/src/svsvllm/app/ui/ui.py)"
        )
        st.markdown(
            "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/svnv-svsv-jm/llm?quickstart=1)"
        )

    return {
        "openai_api_key": openai_api_key,
        "uploaded_files": uploaded_files,
        "selected_language": selected_language,
    }
