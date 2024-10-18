__all__ = ["create_rag_database"]

import typing as ty
from loguru import logger

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from svsvllm.loaders import load_documents


def create_rag_database(
    path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 30,
    model_name: str = "BAAI/bge-base-en-v1.5",
    **kwargs: ty.Any,
) -> FAISS:
    """Create RAG database."""
    documents: ty.List[Document] = load_documents(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs: ty.List[Document] = splitter.split_documents(documents)
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(chunked_docs, embedding=embedder, **kwargs)
    return db
