__all__ = ["clirunner"]

import pytest
from typer.testing import CliRunner


@pytest.fixture
def clirunner() -> CliRunner:
    """App runner. This is needed for testing the main commands."""
    return CliRunner()
