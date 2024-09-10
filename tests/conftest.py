import pytest
import sys, os, pyrootutils
import typing as ty
from loguru import logger

pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)

from _fixtures import *  # pylint: disable=unused-wildcard-import
