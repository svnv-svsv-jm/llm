__all__ = ["NoGPUError", "RetrieverNotInitializedError"]


class NoGPUError(Exception):
    """Exception raised when there is no GPU available."""


class RetrieverNotInitializedError(Exception):
    """Raised when the history-aware retriever is being initialized but the database RAG retriever is not initialized yet."""
