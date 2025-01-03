__all__ = ["NoGPUError", "RetrieverNotInitializedError", "NoChatModelError", "UnsupportedLLMResponse"]


class NoGPUError(Exception):
    """Exception raised when there is no GPU available."""


class RetrieverNotInitializedError(Exception):
    """Raised when the history-aware retriever is being initialized but the database RAG retriever is not initialized yet."""


class NoChatModelError(Exception):
    """Raised when the chat model is not available but it should've been, like when calling `create_agent`."""


class UnsupportedLLMResponse(Exception):
    """When the streaming LLM returns an object that we do not expect.
    This just means that we do not know how to get the LLM's string output out of it, not necessarily that something is wrong.
    """
