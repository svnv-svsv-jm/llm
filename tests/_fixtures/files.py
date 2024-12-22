__all__ = ["uploaded_files"]

import pytest
import uuid
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec


@pytest.fixture
def uploaded_files(request: pytest.FixtureRequest) -> list[UploadedFile]:
    """Fake uploaded files."""
    file_data = getattr(request, "param", dict(name="fake", type="test", data=bytes(1)))
    return [
        UploadedFile(
            record=UploadedFileRec(file_id=str(uuid.uuid4()), **file_data),
            file_urls=None,
        )
    ]
