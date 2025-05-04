import pytest
from streamlit.testing.v1 import AppTest

import svsv


@pytest.fixture
def app() -> AppTest:
    """App."""
    return AppTest.from_function(svsv.run_ui)
