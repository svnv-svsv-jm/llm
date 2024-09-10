import pytest
from unittest.mock import patch
import typing as ty

import torch
from torch.backends import quantized


@pytest.fixture
def patch_torch_quantized_engine(device: torch.device) -> ty.Generator[bool, None, None]:
    """Workaround to try quantization on Mac M1."""
    if device == torch.device("mps"):
        try:
            with patch.object(quantized, "engine", new_value="qnnpack"):
                assert quantized.engine == "qnnpack"
                yield True
        except:
            yield True
    else:
        yield False
