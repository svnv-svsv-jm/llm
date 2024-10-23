import pytest
import sys, os
import typing as ty
from loguru import logger

# Testing environment
# NOTE: The prefix is hardcoded here to avoid importing the settings from the app before the env variables are set...
os.environ["SVSVLLM_TEST_MODE"] = "True"
os.environ["SVSVLLM_UPLOADED_FILES_DIR"] = os.path.join("res", "documents")

import warnings

warnings.filterwarnings("ignore")

import pyrootutils

pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)

# Import fixtures
from _fixtures import *  # pylint: disable=unused-wildcard-import
