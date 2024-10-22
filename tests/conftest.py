import pytest
import sys, os, pyrootutils
import typing as ty
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)

# Testing environment
os.environ["TEST_MODE"] = "True"

# Import fixtures
from _fixtures import *  # pylint: disable=unused-wildcard-import
