__all__ = ["ENV_PREFIX", "LOG_LEVEL_KEY", "DEFAULT_UPLOADED_FILES_DIR"]

import os
from pathlib import Path

from svsvllm.const import *

# Environment prefix for settings
ENV_PREFIX = "SVSVLLM_"

# Logging level key
LOG_LEVEL_KEY = "LOG_LEVEL"

# Default location for documents
DEFAULT_UPLOADED_FILES_DIR = os.path.join(".rag")
Path(DEFAULT_UPLOADED_FILES_DIR).mkdir(parents=True, exist_ok=True)
