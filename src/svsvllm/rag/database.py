__all__ = ["create_rag_database"]

import typing as ty
from loguru import logger

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .documents import load_documents


def create_rag_database(
    path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 30,
    model_name: str = "BAAI/bge-base-en-v1.5",
    **kwargs: ty.Any,
) -> FAISS:
    """Create RAG database.

    Args:
        path (str):
            The path to the directory containing documents to load.

        chunk_size (int, optional):
            Chunk size for `RecursiveCharacterTextSplitter`. Defaults to `512`.

        chunk_overlap (int, optional):
            Chunk overlap for `RecursiveCharacterTextSplitter`. Defaults to `30`.

        model_name (str, optional):
            Chunk embedder model name. Defaults to `"BAAI/bge-base-en-v1.5"`.

    Returns:
        FAISS: The `VectorStore` database.
    """
    logger.trace(f"Creating database from folder: {path}")
    documents: list[Document] = load_documents(path)
    if len(documents) < 1:
        logger.warning(f"No documents found at location: {path}.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs: list[Document] = splitter.split_documents(documents)
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(chunked_docs, embedding=embedder, **kwargs)
    return db
