import pytest
import os
from pathlib import Path


@pytest.fixture
def artifact_location() -> str:
    """Location for test artifacts."""
    loc = "pytest_artifacts"
    Path(loc).mkdir(exist_ok=True, parents=True)
    return loc


@pytest.fixture
def docs_path() -> str:
    """Path to foler with documents."""
    path = os.path.join("res", "documents")
    return path
