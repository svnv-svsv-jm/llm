__all__ = ["load_chat_model"]

import typing as ty
from loguru import logger

import transformers
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.chat_models.mlx import ChatMLX as ChatMLX_
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_core.utils.function_calling import convert_to_openai_tool

from svsvllm.models import load_model
from svsvllm.utils import find_device, add_chat_template, pop_params_not_in_fn
from svsvllm.const import ZEPHYR_CHAT_TEMPLATE as CHAT_TEMPLATE
from svsvllm.types import ModelType, TokenizerType, ChatModelType


# TODO: check if new version fixed this
# NOTE: This code is copy-pasted from `ChatHuggingFace` because the current `langchain_community` version forgot to do it
class ChatMLX(ChatMLX_):
    """We patch the original class by adding the `bind_tools` method.

    This code is copy-pasted from `ChatHuggingFace` because the current `langchain_community` version forgot to do it
    """

    def bind_tools(  # pragma: no cover
        self,
        tools: ty.Sequence[ty.Union[dict[str, ty.Any], type, ty.Callable, BaseTool]],
        *,
        tool_choice: ty.Optional[ty.Union[dict, str, ty.Literal["auto", "none"], bool]] = None,
        **kwargs: ty.Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if len(formatted_tools) != 1:
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, str):
                if tool_choice not in ("auto", "none"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            elif isinstance(tool_choice, bool):
                tool_choice = formatted_tools[0]
            elif isinstance(tool_choice, dict):
                if formatted_tools[0]["function"]["name"] != tool_choice["function"]["name"]:
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tool was {formatted_tools[0]['function']['name']}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. " f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)


def load_hf_pipeline(
    model_name: str,
    device: torch.device,
    chat_template: str | dict[str, ty.Any] | list[dict[str, ty.Any]],
    apply_chat_template: bool = True,
    force_chat_template: bool = False,
    pipeline_kwargs: dict = {},
    **kwargs: ty.Any,
) -> HuggingFacePipeline:
    """Load `HuggingFacePipeline` pipeline."""

    # Load this model
    logger.trace(f"Loading model ({device}): {model_name}")
    hf_model, tokenizer = load_model(model_name, device=device, **kwargs)

    # Apply tokenizer template?
    tokenizer = add_chat_template(
        tokenizer,
        chat_template=chat_template,
        apply_chat_template=apply_chat_template,
        force_chat_template=force_chat_template,
    )

    # Debug
    logger.trace(f"Loaded model: {model_name}")

    # Creating pipeline
    logger.trace(f"Creating `transformers.pipeline`")
    pipeline = transformers.pipeline(
        task="text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map=device,
        **pipeline_kwargs,
    )
    logger.trace(f"Created: {pipeline}")

    # Create HuggingFacePipeline
    logger.trace(f"Creating {HuggingFacePipeline}")
    logger.trace(f"pipeline_kwargs: {pipeline_kwargs}")
    llm = HuggingFacePipeline(pipeline=pipeline, model_id=model_name, pipeline_kwargs=pipeline_kwargs)
    logger.trace(f"Created: {llm}")

    return llm


def load_mlx_pipeline(
    model_name: str,
    device: torch.device,
    chat_template: str | dict[str, ty.Any] | list[dict[str, ty.Any]],
    apply_chat_template: bool = True,
    force_chat_template: bool = False,
    pipeline_kwargs: dict = {},
    **kwargs: ty.Any,
) -> MLXPipeline:
    """Load `MLXPipeline`."""
    # Set up
    logger.trace(f"Loading model ({device}): {model_name}")
    pipeline_kwargs.setdefault("device_map", device)
    _, kwargs = pop_params_not_in_fn(load_model, kwargs)
    # Create MLXPipeline from model ID
    llm = MLXPipeline.from_model_id(
        model_name,
        pipeline_kwargs=pipeline_kwargs,
        **kwargs,
    )
    # Apply tokenizer template?
    llm.tokenizer = add_chat_template(
        llm.tokenizer,
        chat_template=chat_template,
        apply_chat_template=apply_chat_template,
        force_chat_template=force_chat_template,
    )
    logger.trace(f"Created: {llm}")
    return llm


def load_chat_model(
    model_name: str,
    apply_chat_template: bool = False,
    chat_template: str | dict[str, ty.Any] | list[dict[str, ty.Any]] = CHAT_TEMPLATE,
    force_chat_template: bool = False,
    device: torch.device = None,
    pipeline_kwargs: dict[str, ty.Any] | None = None,
    use_mlx: bool = False,
    **kwargs: ty.Any,
) -> tuple[ChatModelType, ModelType, TokenizerType]:
    """Create a chat model.

    Args:
        model_name (str, optional):
            Name of the model to use/download.
            For example: `"meta-llama/Meta-Llama-3.1-8B-Instruct"`.
            For all models, see HuggingFace website.
            Defaults to `None` (chosen from session state).

        apply_chat_template (bool, optional):
            Whether to apply chat template to tokenizer.
            Defaults to `None` (chosen from settings).

        chat_template (str | dict[str, ty.Any] | list[dict[str, ty.Any]]):
            Chat template to enforce when a default one is not available.
            Defaults to `None` (chosen from settings).

        force_chat_template (bool):
            If `True`, the provided chat template will be forced on the tokenizer.
            Defaults to `False`.

        device (torch.device):
            Device. Defaults to `None`.

        pipeline_kwargs (dict, optional):
            See :class:`HuggingFacePipeline`.

        use_mlx (bool):
            Whether to use the `mlx_lm` library or not.
            Defaults to `False`.

        **kwargs (Any):
            See :func:`load_hf_pipeline` or :func:`load_mlx_pipeline`.

    Returns:
        tuple[ChatHuggingFace | ChatMLX, Any, Any]: chat model, plus a reference to the model and tokenzier being used.
    """
    logger.trace("Creating chat model")
    # Sanitize inputs
    if device is None:
        device = find_device()
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    # Load LLM pipeline
    logger.trace("Loading LLM pipeline")
    llm: HuggingFacePipeline | MLXPipeline
    chat_model: ChatHuggingFace | ChatMLX
    if use_mlx:
        llm = load_mlx_pipeline(
            model_name,
            device=device,
            chat_template=chat_template,
            apply_chat_template=apply_chat_template,
            force_chat_template=force_chat_template,
            pipeline_kwargs=pipeline_kwargs,
            **kwargs,
        )
        tokenizer = llm.tokenizer
        model = llm.model
        chat_model = ChatMLX(llm=llm)
    else:
        llm = load_hf_pipeline(
            model_name,
            device=device,
            chat_template=chat_template,
            apply_chat_template=apply_chat_template,
            force_chat_template=force_chat_template,
            pipeline_kwargs=pipeline_kwargs,
            **kwargs,
        )
        tokenizer = llm.pipeline.tokenizer
        model = llm.pipeline.model
        chat_model = ChatHuggingFace(llm=llm, model_id=model_name)

    # Create and save chat model
    logger.trace(f"Chat model created: {chat_model}")

    # NOTE: This is necessary probably due to a bug in `HuggingFacePipeline`
    if isinstance(chat_model, ChatHuggingFace):
        setattr(chat_model, "tokenizer", tokenizer)

    logger.trace(f"Chat model ({type(chat_model)}): {chat_model}")
    logger.trace(f"LLM ({type(llm)}): {llm}")
    logger.trace(f"Model ({type(model)}): {model}")
    logger.trace(f"Tokenizer ({type(tokenizer)}): {tokenizer}")

    return chat_model, model, tokenizer
