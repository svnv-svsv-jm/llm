from unittest import mock

import pydantic
import pytest

from svsv.schema._base import BaseModelWithValidation


def test_schema() -> None:
    """TODO."""
    assert BaseModelWithValidation.is_valid({})

    def raise_error(*args, **kwargs) -> None:  # type: ignore
        raise pydantic.ValidationError(mock.MagicMock(), mock.MagicMock())

    with mock.patch.object(pydantic.BaseModel, "model_validate", side_effect=raise_error):
        assert not BaseModelWithValidation.is_valid({"foo": "bar"})


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
