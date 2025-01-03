__all__ = ["setup_for_streaming", "stream", "stream_open_source_model"]

import typing as ty
from loguru import logger
import streamlit as st
import uuid
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import HumanMessage
from langchain_core.utils.interactive_env import is_interactive_env
from langchain_core.runnables.config import RunnableConfig

from svsvllm.exceptions import RetrieverNotInitializedError
from svsvllm.utils import pop_params_not_in_fn
from svsvllm.chat import extract_message_from_event
from svsvchat.model import create_chat_model
from svsvchat.rag import initialize_rag, create_history_aware_retriever
from svsvchat.agent import create_agent
from svsvchat.settings import settings


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
    try:
        create_history_aware_retriever()
    except RetrieverNotInitializedError as ex:  # pragma: no cover
        if settings.test_mode:
            raise ex

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
        # if i == 0:
        #     yield "\n"
        logger.trace(f"[{i}] Event ({type(event)}): {event}")

        # NOTE: Make sure nothing can happen after each `yield`
        # Thus, we use many nested blocks
        message = extract_message_from_event(event)
        yield message.pretty_repr(html=is_interactive_env())


def stream_open_source_model(
    query: str,
    pipeline_kwargs: dict[str, ty.Any] | None = None,
    agent_executor: CompiledGraph | None = None,
    agent_config: RunnableConfig | None = None,
    setup_kwargs: dict[str, ty.Any] = None,
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
        if setup_kwargs is None:
            setup_kwargs = {}
        agent_executor, agent_config = setup_for_streaming(
            pipeline_kwargs=pipeline_kwargs,
            **setup_kwargs,
        )
    # Stream
    yield from stream(
        query,
        agent_executor=agent_executor,
        agent_config=agent_config,
        **kwargs,
    )
