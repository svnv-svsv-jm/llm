import pytest
from loguru import logger
import typing as ty
import sys

import torch
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from svsvllm.utils.accelerators import find_device


class DeepEvalLLM(DeepEvalBaseLLM):
    """Custom evaluator LLM for `deepeval`."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        accelerator: str = "auto",
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            model (AutoModelForCausalLM):
                Input LLM model.

            tokenizer (AutoTokenizer):
                Input tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = find_device(accelerator)
        super().__init__(**kwargs)

    @property
    def __name__(self) -> str:
        return f"{self.model.__class__.__name__}"

    def load_model(self) -> AutoModelForCausalLM:
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        model.to(self.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return f"{self.tokenizer.batch_decode(generated_ids)[0]}"

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.__name__


def test_llm(mistral_small: tuple[AutoModelForCausalLM, AutoTokenizer]) -> None:
    """Test usage of `deepeval`."""
    # Evaluator
    model, tokenizer = mistral_small
    evaluator = Mistral7B(model=model, tokenizer=tokenizer)

    # Test case
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        expected_output="You're eligible for a 30 day refund at no extra cost.",
        actual_output="We offer a 30-day full refund at no extra cost.",
        context=["All customers are eligible for a 30 day full refund at no extra cost."],
        retrieval_context=["Only shoes can be refunded."],
        tools_called=["WebSearch"],
        expected_tools=["WebSearch", "QueryDatabase"],
    )

    # Test
    assert_test(test_case, [AnswerRelevancyMetric(model=evaluator)])


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
