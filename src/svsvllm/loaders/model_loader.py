__all__ = ["load_model"]

import os
import typing as ty
from loguru import logger

import torch
from torch.quantization import quantize_dynamic
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from optimum.quanto import QuantizedModelForCausalLM, qint4


def load_model(
    model_name: str,
    bnb_config: BitsAndBytesConfig | None = None,
    quantize: bool = False,
    quantize_w_torch: bool = False,
    device: torch.device = None,
    token: str | None = None,
    revision: str = "float16",
) -> ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """Helper to load and quantize a model."""
    logger.debug(f"Loading model '{model_name}'...")

    # Token
    if token is None:
        token = os.environ["HUGGINGFACE_TOKEN"]

    # Load model
    def load() -> AutoModelForCausalLM:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            token=token,
            revision=revision,
            device_map=device,
        )

    if quantize and not quantize_w_torch:
        try:  # We try to get a previously quantized and saved model
            model = QuantizedModelForCausalLM.from_pretrained(f"models/{model_name}")
            already_quantoed = True
        except:
            model = load()
            already_quantoed = False
    else:
        model = load()
    logger.debug(f"Loaded model '{model}'...")

    # Quantize
    if quantize:
        if quantize_w_torch:
            logger.debug("Quantizing with `torch.quantization.quantize_dynamic`...")
            model = quantize_dynamic(model, dtype=torch.qint8)
        else:
            if not already_quantoed:
                logger.debug(f"Quantizing with {QuantizedModelForCausalLM}...")
                model = QuantizedModelForCausalLM.quantize(model, weights=qint4, exclude="lm_head")
                logger.debug("Saving pretrained quantized model.")
                model.save_pretrained(f"models/{model_name}")
        logger.debug(f"Quantized {model_name}")
    if isinstance(model, QuantizedModelForCausalLM):
        model = model._wrapped

    # Tokenizer
    logger.debug(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    logger.debug(f"Loaded tokenizer '{tokenizer}'...")
    return model, tokenizer
