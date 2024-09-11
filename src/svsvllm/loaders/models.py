__all__ = ["load_model"]

import os
import typing as ty
from loguru import logger

import torch
from torch.quantization import quantize_dynamic, get_default_qconfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from optimum.quanto import QuantizedModelForCausalLM, qint4

from svsvllm.utils import CommandTimer, get_default_backend

DEFAULT_BACKEND = get_default_backend()


def load_model(
    model_name: str,
    bnb_config: BitsAndBytesConfig | None = None,
    quantize: bool = False,
    quantize_w_torch: bool = False,
    device: torch.device = None,
    token: str | None = None,
    revision: str | None = None,
    model_class: ty.Type[AutoModelForCausalLM] | None = None,
    tokenizer_class: ty.Type[AutoTokenizer] | None = None,
    backend: str = DEFAULT_BACKEND,
) -> ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """Load LLM.

    Args:
        model_name (str):
            Name of the model to use/download. For example: `"meta-llama/Meta-Llama-3.1-8B-Instruct"`.
            For all models, see HuggingFace website.

        bnb_config (BitsAndBytesConfig | None, optional):
            Configuration for quantization. This only works on CUDA.
            Defaults to `None`.

        quantize (bool, optional):
            Whether to quantize or not. Defaults to `False`.

        quantize_w_torch (bool, optional):
            If `True`, quantization will happen using `from torch.quantization import quantize_dynamic, get_default_qconfig`.
            If `False`, quantization will happen using `from optimum.quanto import QuantizedModelForCausalLM`.
            Defaults to `False`.

        device (torch.device, optional):
            Accelarator. Defaults to `None`.

        token (str | None, optional):
            HuggingFace token, necessary to download models.
            Defaults to `None`, meaning that it will be read from the environment variables `HUGGINGFACE_TOKEN`.

        revision (str, optional):
            For example: `"float16"`. Defaults to `None`.

        model_class (ty.Type[AutoModelForCausalLM], optional):
            Defaults to `AutoModelForCausalLM`.

        tokenizer_class (ty.Type[AutoTokenizer], optional):
            Defaults to `AutoTokenizer`.

        backend (str):
            A string representing the target backend.
            Currently supports `x86`, `fbgemm`, `qnnpack` and `onednn`.

    Returns:
        ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
            Loaded LLM and its corresponding tokenizer.
    """

    logger.debug(f"Loading model '{model_name}'...")

    # HuggingFace token or it won't download
    if token is None:
        token = os.environ["HUGGINGFACE_TOKEN"]

    # Sanitize inputs
    if model_class is None:
        model_class = AutoModelForCausalLM
    if tokenizer_class is None:
        tokenizer_class = AutoTokenizer

    # Tokenizer
    logger.debug(f"Loading tokenizer...")
    tokenizer = tokenizer_class.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token,
        # use_fast=False,
    )
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
            # Also see: https://github.com/pytorch/pytorch/issues/123507
            logger.debug("Quantizing with `torch.quantization.quantize_dynamic`...")
            # Define the qconfig (using 'fbgemm' or 'qnnpack' configuration)
            qconfig = get_default_qconfig(backend)
            qconfig_spec = {torch.nn.Module: qconfig}
            logger.debug(f"Configuration: {qconfig_spec}")
            # Apply the qconfig to the model
            # model.qconfig = qconfig
            model = quantize_dynamic(
                model,
                dtype=torch.qint8,
                qconfig_spec=qconfig_spec,
            )
        else:
            if not already_quantoed:
                logger.debug(f"Quantizing with {QuantizedModelForCausalLM}...")
                model = QuantizedModelForCausalLM.quantize(model, weights=qint4, exclude="lm_head")
                logger.debug("Saving pretrained quantized model.")
                model.save_pretrained(f"models/{model_name}")
        logger.debug(f"Quantized {model_name}")
    if isinstance(model, QuantizedModelForCausalLM):
        model = model._wrapped

    # Return model and tokenizer
    return model, tokenizer
