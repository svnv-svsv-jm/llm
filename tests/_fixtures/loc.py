__all__ = ["artifact_location", "docs_path"]

import pytest
import os
from pathlib import Path


@pytest.fixture(scope="session")
def artifact_location() -> str:
    """Location for test artifacts."""
    loc = "pytest_artifacts"
    Path(loc).mkdir(exist_ok=True, parents=True)
    return loc


@pytest.fixture(scope="session")
def docs_path() -> str:
    """Path to foler with documents."""
    path = os.path.join("res", "documents")
    return path
