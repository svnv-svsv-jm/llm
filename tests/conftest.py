import pytest
import sys, os
import typing as ty
from loguru import logger as loguru_logger
import warnings
from pathlib import Path
import shutil
import pyrootutils

warnings.filterwarnings("ignore")
pyrootutils.setup_root(search_from=".", pythonpath=True, dotenv=True, cwd=True)


# Import fixtures
from _fixtures import *  # pylint: disable=unused-wildcard-import


@pytest.fixture(autouse=True)
def setup(delete_all_files_in_rag: None) -> ty.Iterator[None]:
    """Set up and clean up."""
    yield
