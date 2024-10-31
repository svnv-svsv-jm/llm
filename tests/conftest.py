import pytest
import sys, os
import typing as ty
from loguru import logger
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
