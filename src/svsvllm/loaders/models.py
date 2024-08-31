__all__ = ["load_model"]

import os
import typing as ty
from loguru import logger

import torch
from torch.quantization import quantize_dynamic
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from optimum.quanto import QuantizedModelForCausalLM, qint4

from svsvllm.utils import CommandTimer


def load_model(
    model_name: str,
    bnb_config: BitsAndBytesConfig | None = None,
    quantize: bool = False,
    quantize_w_torch: bool = False,
    device: torch.device = None,
    token: str | None = None,
    revision: str | None = None,
    model_class: ty.Type[AutoModelForCausalLM] = AutoModelForCausalLM,
    tokenizer_class: ty.Type[AutoTokenizer] = AutoTokenizer,
) -> ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """Load LLM.

    Args:
        model_name (str): _description_

        bnb_config (BitsAndBytesConfig | None, optional): _description_. Defaults to `None`.

        quantize (bool, optional): _description_. Defaults to `False`.

        quantize_w_torch (bool, optional): _description_. Defaults to `False`.

        device (torch.device, optional): _description_. Defaults to `None`.

        token (str | None, optional): _description_. Defaults to `None`.

        revision (str, optional): For example: `"float16"`. Defaults to `None`.

        model_class (ty.Type[AutoModelForCausalLM], optional): _description_. Defaults to `AutoModelForCausalLM`.

        tokenizer_class (ty.Type[AutoTokenizer], optional): _description_. Defaults to `AutoTokenizer`.

    Returns:
        ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]: _description_
    """

    logger.debug(f"Loading model '{model_name}'...")

    # HuggingFace token or it won't download
    if token is None:
        token = os.environ["HUGGINGFACE_TOKEN"]

    # Tokenizer
    logger.debug(f"Loading tokenizer...")
    tokenizer = tokenizer_class.from_pretrained(model_name, trust_remote_code=True, token=token)
    logger.debug(f"Loaded tokenizer '{tokenizer}'...")

    # Load model
    def load() -> AutoModelForCausalLM:
        with CommandTimer(f"Model loading: {model_name}"):
            model = model_class.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                token=token,
                revision=revision,
                device_map=device,
            )
        return model

    if quantize and not quantize_w_torch:
        try:  # We try to get a previously quantized and saved model
            logger.trace(f"Trying to load {model_name} using {QuantizedModelForCausalLM}...")
            model = QuantizedModelForCausalLM.from_pretrained(f"models/{model_name}")
            already_quantoed = True
            logger.trace(f"Success!")
        except:
            logger.trace(f"Could not load {model_name} using {QuantizedModelForCausalLM}.")
            model = load()
            already_quantoed = False
    else:
        logger.debug(f"Loading {model_name} without quantizing...")
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

    return model, tokenizer
