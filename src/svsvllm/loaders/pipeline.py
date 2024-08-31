__all__ = ["pipeline"]

import typing as ty
from loguru import logger
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as _pipeline
from optimum.quanto import QuantizedModelForCausalLM


def pipeline(
    model: AutoModelForCausalLM | QuantizedModelForCausalLM | None = None,
    tokenizer: AutoTokenizer | None = None,
    device: torch.device | None = None,
    temperature: float = 0.2,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
    return_full_text: bool = False,
    max_new_tokens: int = 500,
    task: str = "text-generation",
    **kwargs: ty.Any,
) -> HuggingFacePipeline:
    """Cerbero pipeline."""
    logger.debug("Pipeline...")
    # with patch.object(QuantizedModelForCausalLM, "__module__", return_value="torch"):
    pipe = _pipeline(
        model=model,
        tokenizer=tokenizer,
        task=task,
        device=device,
        temperature=temperature,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        return_full_text=return_full_text,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    logger.debug(f"Pipeline: {llm}")
    return llm
