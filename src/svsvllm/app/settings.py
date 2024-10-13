__all__ = ["TEST_MODE"]

import os

TEST_MODE = os.environ.get("TEST_MODE", "False").lower() == "true"
