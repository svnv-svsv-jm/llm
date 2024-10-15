__all__ = ["NoGPUError"]


class NoGPUError(Exception):
    """Exception raised when there is no GPU available."""
