__all__ = ["initialize_rag"]

from loguru import logger


def initialize_rag(force_recreate: bool = False) -> None:
    """Initializes the RAG."""
    logger.trace("Initializing RAG")
    logger.trace("RAG initialied")
