import pytest
import typing as ty

import svsv


@pytest.fixture
def settings() -> ty.Iterator[svsv.Settings]:
    """App's settings."""
    yield svsv.settings
