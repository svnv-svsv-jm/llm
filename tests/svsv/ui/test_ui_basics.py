import pytest
from streamlit.testing.v1 import AppTest


def test_ui_basics(app: AppTest) -> None:
    """Run UI function via app and check no exceptions are raised."""
    app.run()
    assert not app.exception


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
