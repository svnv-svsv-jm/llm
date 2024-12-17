__all__ = [
    "get_openai_response",
    "setup_for_streaming",
    "stream",
    "invoke",
    "stream_open_source_model",
    "invoke_open_source_model",
]

import typing as ty
from loguru import logger
import streamlit as st
import uuid
from langgraph.graph.graph import CompiledGraph
from langgraph.pregel.io import AddableValuesDict
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.utils.interactive_env import is_interactive_env
from langchain_core.runnables.config import RunnableConfig
from openai import OpenAI

from svsvllm.schema import ChatMLXEvent
from svsvllm.utils import pop_params_not_in_fn
from .session_state import session_state
from .model import create_chat_model
from .rag import initialize_rag, create_history_aware_retriever
from .agent import create_agent


def get_openai_response(openai_api_key: str, model: str) -> str:
    """Returns the response from the OpenAI model.

    The full chat history up to now is fed as input.

    Args:
        openai_api_key (str):
            OpenAI API key.

        model (str):
            OpenAI model name.

    Returns:
        str: LLM's response as text.
    """
    state = session_state
    # OpenAI client
    if "openai_client" not in st.session_state:
        logger.trace(f"Creating new OpenAI client.")
        client = OpenAI(api_key=openai_api_key)
        st.session_state["openai_client"] = client
        logger.trace(f"Created new OpenAI client.")
    else:
        logger.trace(f"Getting OpenAI client from session state.")
        client = st.session_state["openai_client"]
        logger.trace(f"Got OpenAI client from session state.")

    # Create response
    logger.trace(f"Calling `OpenAI.chat.completions.create`")
    response = client.chat.completions.create(
        model=model,
        # NOTE: This param is type-ignored because even though the type is wrong, all that matters is that the provided objects (in the list) implement the required class properties: content, role, name
        messages=state.messages,  # type: ignore
    )
    msg = response.choices[0].message.content

    # Sanity check and return
    if msg is None:
        msg = ""
    return f"{msg}"


def setup(
    model_name: str = None,
    pipeline_kwargs: dict[str, ty.Any] | None = None,
    **kwargs: ty.Any,
) -> CompiledGraph:
    """General setup.

    Args:
        model_name (str, optional):
            Name of the model to use/download.
            For example: `"meta-llama/Meta-Llama-3.1-8B-Instruct"`.
            For all models, see HuggingFace website.
            Defaults to `None` (chosen from session state).

        pipeline_kwargs (dict, optional):
            See :class:`HuggingFacePipeline`.

    Returns:
        CompiledGraph: LLM agent.
    """
    # Initialize model
    logger.trace(f"Initializing model")
    create_chat_model(model_name=model_name, pipeline_kwargs=pipeline_kwargs, **kwargs)

    # Initialize RAG
    logger.trace(f"Initializing RAG and history-aware retriever")
    initialize_rag()
    create_history_aware_retriever()

    # Create agent
    logger.trace(f"Initializing agent")
    agent_executor = create_agent()
    return agent_executor


def setup_for_streaming(**kwargs: ty.Any) -> tuple[CompiledGraph, RunnableConfig]:
    """Sets up all that is needed to have an agent ready to stream.

    Args:
        **kwargs (Any):
            See :func:`setup`.

    Returns:
        agent_executor (CompiledGraph):
            Agent.

        agent_config (RunnableConfig):
            Agent's configuration.
    """
    agent_executor = setup(**kwargs)

    # Create chat configuration
    logger.trace("Creating chat configuration")
    # Generate thread ID if not exists, or use the one in the session state
    thread_id = st.session_state.get("thread_id", None)
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        st.session_state["thread_id"] = thread_id
    logger.trace(f"Using thread ID: {thread_id}")

    # Create configuration
    agent_config = RunnableConfig(**{"configurable": {"thread_id": thread_id}})
    st.session_state["agent_config"] = agent_config
    logger.trace(f"Using agent config: {agent_config}")

    # Return
    return agent_executor, agent_config


def invoke(
    query: str,
    agent_executor: CompiledGraph,
    agent_config: RunnableConfig,
    **kwargs: ty.Any,
) -> str | ty.Any:
    """Invoke LLM.

    Args:
        query (str):
            Input query.

        agent_executor (CompiledGraph, optional):
            Agent. This is created internally when not provided.

        agent_config (RunnableConfig, optional):
            Agent's configuration. This is created internally when not provided.

        **kwargs:
            See `langgraph.graph.graph.CompiledGraph.invoke()`.

    Yields:
        str | Any: LLM's response.
    """
    # Pop params
    logger.trace(f"Kwargs: {kwargs}")
    kwargs, _ = pop_params_not_in_fn(agent_executor.invoke, params=kwargs)
    logger.trace(f"Kwargs: {kwargs}")
    # Invoke
    event = agent_executor.invoke({"messages": [HumanMessage(content=query)]}, config=agent_config)
    logger.trace(f"Event ({type(event)}): {event}")
    # If not a `dict`, yield it
    if not isinstance(event, dict):
        return event
    if ChatMLXEvent.is_valid(event):
        logger.trace("ChatMLXEvent")
        ev = ChatMLXEvent(**event)
        out = ev.payload.result  # pylint: disable=no-member
        logger.trace(out)
        return out
    # Get the messages
    messages: list[BaseMessage] = event.get("messages", None)
    logger.trace(f"messages ({type(messages)}): {messages}")
    # Make sure nothing can happen after each `yield`
    if isinstance(messages, list):
        if len(messages) > 0:
            last_message: BaseMessage = messages[-1]
            if isinstance(last_message, BaseMessage):
                return last_message.pretty_repr(html=is_interactive_env())
            return last_message
        return messages
    return messages


def stream(
    query: str,
    agent_executor: CompiledGraph,
    agent_config: RunnableConfig,
    **kwargs: ty.Any,
) -> ty.Iterator[str | ty.Any]:
    """Stream response from agent.

    Args:
        query (str):
            Input query.

        agent_executor (CompiledGraph, optional):
            Agent. This is created internally when not provided.

        agent_config (RunnableConfig, optional):
            Agent's configuration. This is created internally when not provided.

        **kwargs:
            See `langgraph.graph.graph.CompiledGraph.stream()`.

    Yields:
        str | Any: LLM's response.
    """
    # Stream
    logger.trace(f"Streaming")
    kwargs.setdefault("stream_mode", "values")

    # Pop params
    logger.trace(f"Kwargs: {kwargs}")
    kwargs, _ = pop_params_not_in_fn(agent_executor.stream, params=kwargs)
    logger.trace(f"Kwargs: {kwargs}")

    # NOTE: Stream yields events like: `{'alist': ['Ex for stream_mode="values"'], 'another_list': []}`
    for i, event in enumerate(  # pylint: disable=too-many-nested-blocks
        agent_executor.stream(
            {"messages": [HumanMessage(content=query)]},
            config=agent_config,
            **kwargs,
        )
    ):
        if i == 0:
            yield "\n"
        logger.trace(f"Event ({type(event)}): {event}")
        # If not a `dict`, yield it
        if isinstance(event, AddableValuesDict):
            messages: list[BaseMessage] = event["messages"]
            msg = messages[-1]
            yield msg.content
        elif not isinstance(event, dict):
            yield event
        else:
            if ChatMLXEvent.is_valid(event):
                logger.trace("ChatMLXEvent")
                ev = ChatMLXEvent(**event)
                out = ev.payload.result  # pylint: disable=no-member
                logger.trace(out)
                message = out[-1][-1][-1]
                yield message.content
            else:
                # Get the messages
                messages = event.get("messages", None)
                logger.trace(f"messages ({type(messages)}): {messages}")
                # Make sure nothing can happen after each `yield`
                if isinstance(messages, list):
                    if len(messages) > 0:
                        last_message: BaseMessage = messages[-1]
                        if isinstance(last_message, BaseMessage):
                            yield last_message.pretty_repr(html=is_interactive_env())
                        else:
                            yield last_message
                    else:
                        yield messages
                else:
                    yield messages


def stream_open_source_model(
    query: str,
    pipeline_kwargs: dict[str, ty.Any] | None = None,
    agent_executor: CompiledGraph | None = None,
    agent_config: RunnableConfig | None = None,
    **kwargs: ty.Any,
) -> ty.Iterator[str | ty.Any]:
    """Stream response (given input query) from the open-source LLM.

    Args:
        query (str):
            Input query.

        pipeline_kwargs (dict, optional):
            See :class:`HuggingFacePipeline`.

        agent_executor (CompiledGraph, optional):
            Agent. This is created internally when not provided.

        agent_config (RunnableConfig, optional):
            Agent's configuration. This is created internally when not provided.

        **kwargs:
            See `langgraph.graph.graph.CompiledGraph.stream()`.

    Yields:
        str | Any: LLM's response.
    """
    # Create agent
    if agent_executor is None or agent_config is None:
        agent_executor, agent_config = setup_for_streaming(pipeline_kwargs=pipeline_kwargs)
    # Stream
    yield from stream(query, agent_executor=agent_executor, agent_config=agent_config, **kwargs)


def invoke_open_source_model(
    query: str,
    pipeline_kwargs: dict[str, ty.Any] | None = None,
    agent_executor: CompiledGraph | None = None,
    agent_config: RunnableConfig | None = None,
    **kwargs: ty.Any,
) -> str | ty.Any:
    """Stream response (given input query) from the open-source LLM.

    Args:
        query (str):
            Input query.

        pipeline_kwargs (dict, optional):
            See :class:`HuggingFacePipeline`.

        agent_executor (CompiledGraph, optional):
            Agent. This is created internally when not provided.

        agent_config (RunnableConfig, optional):
            Agent's configuration. This is created internally when not provided.

        **kwargs:
            See `langgraph.graph.graph.CompiledGraph.invoke()`.

    Returns:
        str | Any: LLM's response.
    """
    # Create agent
    if agent_executor is None or agent_config is None:
        agent_executor, agent_config = setup_for_streaming(pipeline_kwargs=pipeline_kwargs)
    # Stream
    return invoke(query, agent_executor=agent_executor, agent_config=agent_config, **kwargs)
