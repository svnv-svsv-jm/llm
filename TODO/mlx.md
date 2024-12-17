# [ ] mlx-llm & mlx-lm

**Status:** To do  
**Priority:** Medium  
**Description:**

Check this one out:

- [mlx-llm](https://github.com/riccardomusmeci/mlx-llm/tree/main)
- [mlx (ml-explore)](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [llm-mlc](https://llm.mlc.ai/docs/get_started/introduction.html).
- [mlx-llm+langchain](https://python.langchain.com/docs/integrations/chat/mlx/)

Libraries to load and use LLMs on Mac M1. They also have their collections of models.

Also think about wrapping it:

```python
import torch
import torch.nn as nn

class MLXModelWrapper(nn.Module):
    def __init__(self, mlx_model):
        super().__init__()
        self.mlx_model = mlx_model  # Store the mlx model

    def forward(self, *args, **kwargs):
        # Here we assume your mlx_model has a `predict` method
        # Adjust to match your mlx model's actual API

        # Convert PyTorch tensors to numpy if necessary
        inputs = [arg.cpu().detach().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]
        mlx_output = self.mlx_model.predict(*inputs, **kwargs)

        # Convert the mlx output back to a PyTorch tensor
        # Assuming mlx_output is a numpy array; adjust if otherwise
        return torch.tensor(mlx_output)

# Example of initializing with your mlx model
# Assuming you have an mlx_model instance
mlx_model = ...  # Your MLX model instance
wrapped_model = MLXModelWrapper(mlx_model)

# Now you can use it with transformers.pipeline
from transformers import pipeline

# Assuming the pipeline task matches your model's output and input requirements
pipe = pipeline("text-classification", model=wrapped_model, tokenizer="bert-base-uncased")

# Now the pipeline should treat your MLX model as a PyTorch model
result = pipe("Example input text")
print(result)
```
