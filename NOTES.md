# Notes

## Quantization

For MPS support, see [Optimum Quanto](https://github.com/huggingface/optimum-quanto).

Maybe also try:

```python
from torch.backends import quantized
from unittest.mock import patch

@patch.object(quantized, "engine", new="qnnpack")
```
