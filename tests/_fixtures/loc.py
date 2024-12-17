__all__ = ["artifacts_location", "docs_path", "res_docs_path"]

import pytest
import os
from pathlib import Path


@pytest.fixture(scope="session")
def artifacts_location() -> str:
    """Location for test artifacts."""
    loc = "pytest_artifacts"
    Path(loc).mkdir(exist_ok=True, parents=True)
    return loc


@pytest.fixture(scope="session")
def docs_path() -> str:
    """Path to folder with documents."""
    path = os.path.join("res", "documents")
    return path


@pytest.fixture(scope="session")
def res_docs_path() -> str:
    """Path to test folder with documents."""
    path = os.path.join("tests", "res", "rag")
    return path
