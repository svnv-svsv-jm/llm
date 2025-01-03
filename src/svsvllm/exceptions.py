__all__ = ["NoGPUError", "RetrieverNotInitializedError", "NoChatModelError"]


class NoGPUError(Exception):
    """Exception raised when there is no GPU available."""


class RetrieverNotInitializedError(Exception):
    """Raised when the history-aware retriever is being initialized but the database RAG retriever is not initialized yet."""


class NoChatModelError(Exception):
    """Raised when the chat model is not available but it should've been, like when calling `create_agent`."""
