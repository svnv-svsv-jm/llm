import pytest
from unittest.mock import MagicMock
import typing as ty
import warnings
import pyrootutils

warnings.filterwarnings("ignore")
pyrootutils.setup_root(search_from=".", pythonpath=True, dotenv=True, cwd=True)


# Import fixtures
from _fixtures import *  # pylint: disable=unused-wildcard-import


@pytest.fixture(autouse=True)
def setup(
    log_to_file: int,
    copy_res_docs_to_rag: None,
) -> ty.Iterator[None]:
    """Set up and clean up."""
    yield
