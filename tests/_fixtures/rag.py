import pytest
import typing as ty
from loguru import logger

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS

from svsvllm.loaders import load_documents
from svsvllm.rag import create_rag_database


@pytest.fixture(scope="session")
def documents(docs_path: str) -> ty.List[Document]:
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
