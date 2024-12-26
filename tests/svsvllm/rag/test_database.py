import pytest
from loguru import logger
import typing as ty
import sys, os

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStoreRetriever


def test_create_rag_database(database: FAISS, retriever: VectorStoreRetriever) -> None:
    """Test: `create_rag_database` is used in the fixtures. This test is minimal."""
    logger.info(f"Database: {database}")
    logger.info(f"Retriever: {retriever.metadata}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
