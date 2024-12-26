__all__ = ["load_model", "load_tokenizer"]

import os
import typing as ty
from loguru import logger

import torch
from torch.quantization import quantize_dynamic, get_default_qconfig  # type: ignore
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from optimum.quanto import QuantizedModelForCausalLM, qint4
import huggingface_hub
from pathlib import Path
from codetiming import Timer as CommandTimer

from svsvllm.const import HUGGINFACE_TOKEN_KEY
from svsvllm.utils import get_default_backend

DEFAULT_BACKEND = get_default_backend()


def _handle_inputs(
    model_name: str,
    token: str | None,
    model_class: ty.Type[AutoModelForCausalLM] | None,
    tokenizer_class: ty.Type[AutoTokenizer] | None,
    models_dir: str,
) -> tuple[str, type[AutoModelForCausalLM], type[AutoTokenizer]]:
    """Handle inputs."""
    # Models directory
    savedir = os.path.join(f"{models_dir}", f"{model_name}")
    Path(savedir).mkdir(exist_ok=True, parents=True)

    # HuggingFace token or it won't download
    if token is None:
        token = os.environ[HUGGINFACE_TOKEN_KEY]
    huggingface_hub.login(token)

    # Sanitize inputs
    if model_class is None:
        model_class = AutoModelForCausalLM
    if tokenizer_class is None:
        tokenizer_class = AutoTokenizer

    return savedir, model_class, tokenizer_class


def load_tokenizer(
    model_name: str,
    token: str | None = None,
    model_class: ty.Type[AutoModelForCausalLM] | None = None,
    tokenizer_class: ty.Type[AutoTokenizer] | None = None,
    models_dir: str = ".models",
) -> PreTrainedTokenizerBase:
    """Load tokenizer.

    Args:
        model_name (str):
            Name of the model to use/download. For example: `"meta-llama/Meta-Llama-3.1-8B-Instruct"`.
            For all models, see HuggingFace website.
        token (str | None, optional):
            HuggingFace token, necessary to download models.
            Defaults to `None`, meaning that it will be read from the environment variables `HF_TOKEN`.

        model_class (ty.Type[AutoModelForCausalLM], optional):
            Defaults to `AutoModelForCausalLM`.

        tokenizer_class (ty.Type[AutoTokenizer], optional):
            Defaults to `AutoTokenizer`.

        models_dir (str):
            Directory where to save quantized models.
            Defaults to `".models"`.

    Returns:
        PreTrainedTokenizerBase: _description_
    """
    logger.debug(f"Loading model '{model_name}'...")

    savedir, model_class, tokenizer_class = _handle_inputs(
        model_name,
        token=token,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        models_dir=models_dir,
    )

    # Tokenizer
    logger.debug(f"Loading tokenizer...")
    tokenizer = tokenizer_class.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token,
        # use_fast=False,
    )
    logger.debug(f"Loaded tokenizer ({type(tokenizer)}) '{model_name}'...")
    return tokenizer


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
    models_dir: str = ".models",
    load_in_4bit: bool = False,
) -> ty.Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
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
            Defaults to `None`, meaning that it will be read from the environment variables `HF_TOKEN`.

        revision (str, optional):
            For example: `"float16"`. Defaults to `None`.

        model_class (ty.Type[AutoModelForCausalLM], optional):
            Defaults to `AutoModelForCausalLM`.

        tokenizer_class (ty.Type[AutoTokenizer], optional):
            Defaults to `AutoTokenizer`.

        backend (str):
            A string representing the target backend.
            Currently supports `x86`, `fbgemm`, `qnnpack` and `onednn`.

        models_dir (str):
            Directory where to save quantized models.
            Defaults to `".models"`.

        load_in_4bit (bool):
            Whether to try to load an already quantized model.
            Defaults to `False`.

    Returns:
        ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
            Loaded LLM and its corresponding tokenizer.
    """
    logger.debug(f"Loading model '{model_name}'...")

    savedir, model_class, tokenizer_class = _handle_inputs(
        model_name,
        token=token,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        models_dir=models_dir,
    )

    tokenizer = load_tokenizer(
        model_name,
        token=token,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        models_dir=models_dir,
    )

    # Load model
    def load() -> AutoModelForCausalLM:
        """Load model."""
        with CommandTimer(f"Model loading: {model_name}"):
            model = model_class.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                token=token,
                revision=revision,
                device_map=device,
                load_in_4bit=load_in_4bit,
            )
        return model

    # Quantization settings
    already_quantoed = False  # Needed for later
    if quantize and not quantize_w_torch:
        try:  # We try to get a previously quantized and saved model
            logger.trace(f"Trying to load {model_name} using {QuantizedModelForCausalLM}...")
            model = QuantizedModelForCausalLM.from_pretrained(savedir)
            already_quantoed = True
            logger.trace(f"Success!")
        except:  # pragma: no cover
            logger.trace(f"Could not load {model_name} using {QuantizedModelForCausalLM}.")
            model = load()
    else:
        logger.debug(f"Loading {model_name}...")
        model = load()
    logger.debug(f"Loaded model ({type(model)}) '{model_name}'...")

    # Quantize
    if quantize:
        # Quantize with PyTorch
        if quantize_w_torch:
            # Also see: https://github.com/pytorch/pytorch/issues/123507
            logger.debug(f"Quantizing {model_name} with `torch.quantization.quantize_dynamic`...")
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
        # Quantize with `optimum.quanto`
        else:
            if not already_quantoed:
                logger.debug(f"Quantizing {model_name} with {QuantizedModelForCausalLM}...")
                model = QuantizedModelForCausalLM.quantize(model, weights=qint4, exclude="lm_head")
                logger.debug("Saving pretrained quantized model.")
                model.save_pretrained(savedir)
        logger.debug(f"Quantized {model_name}")
    # Make sure to convert back to `PreTrainedModel`
    if isinstance(model, QuantizedModelForCausalLM):
        model = model._wrapped

    if device is not None:
        logger.trace(f"Moving model to {device}")
        model = model.to(device)

    # Return model and tokenizer
    logger.trace(f"Returning model ({type(model)}) and tokenizer ({type(tokenizer)}).")
    return model, tokenizer
