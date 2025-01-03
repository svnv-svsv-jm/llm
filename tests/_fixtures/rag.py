__all__ = ["delete_all_files_in_rag", "copy_res_docs_to_rag", "documents", "database", "retriever"]

import pytest
import os
import typing as ty
from loguru import logger
import shutil
from pathlib import Path

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS

from svsvllm.rag import create_rag_database, load_documents
from svsvchat.settings import Settings


@pytest.fixture
def delete_all_files_in_rag(settings: Settings) -> None:
    """Deletes all files in RAG folder."""
    uploaded_files_dir = os.path.abspath(settings.uploaded_files_dir)
    if Path(uploaded_files_dir).is_dir():
        logger.debug(f"Deleting: {uploaded_files_dir}")
        shutil.rmtree(uploaded_files_dir)
    Path(uploaded_files_dir).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def copy_res_docs_to_rag(
    delete_all_files_in_rag: None,
    res_docs_path: str,
    settings: Settings,
) -> None:
    """Copy test resource documents to RAG folder."""
    uploaded_files_dir = os.path.abspath(settings.uploaded_files_dir)
    shutil.copytree(res_docs_path, uploaded_files_dir, dirs_exist_ok=True)


@pytest.fixture(scope="session")
def documents(docs_path: str) -> list[Document]:
    """Loaded documents."""
    docs: ty.List[Document] = load_documents(docs_path)
    return docs


@pytest.fixture(scope="session")
def database(docs_path: str) -> FAISS:
    """Database for the RAG."""
    logger.debug("Database for documents...")
    db = create_rag_database(docs_path)
    return db


@pytest.fixture(scope="session")
def retriever(database: FAISS) -> VectorStoreRetriever:
    """`VectorStoreRetriever` object."""
    logger.debug("Retriever...")
    retriever = database.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever
