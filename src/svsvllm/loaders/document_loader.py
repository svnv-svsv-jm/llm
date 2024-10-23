__all__ = ["load_documents"]

import os
from typing import List
from loguru import logger
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document


def load_documents(path: str) -> List[Document]:
    """Loads documents from the specified directory path.

    This function supports loading of PDF, Markdown, and HTML documents by utilizing
    different loaders for each file type. It checks if the provided path exists and
    raises a `FileNotFoundError` if it does not. It then iterates over the supported
    file types and uses the corresponding loader to load the documents into a list.

    Args:
        path (str): The path to the directory containing documents to load.

    Returns:
        List[Document]: A list of loaded documents.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    # Sanitize input
    path = f"{path}"

    # Raise error if path does not exist
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    # Create loaders
    loaders = {
        ".pdf": DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,  # type: ignore
            show_progress=True,
            use_multithreading=True,
        ),
        ".md": DirectoryLoader(
            path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
        ),
        ".txt": DirectoryLoader(
            path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
        ),
    }

    # Load docs
    docs = []
    for file_type, loader in loaders.items():
        logger.info(f"Loading {file_type} files from {path}...")
        docs.extend(loader.load())
    return docs
