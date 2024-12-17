__all__ = ["uuid_str"]

import uuid


def uuid_str() -> str:
    """Create a `str` UUID."""
    return str(uuid.uuid4())
