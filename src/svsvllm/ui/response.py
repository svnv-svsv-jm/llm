__all__ = ["get_openai_response", "setup_for_streaming", "stream", "get_response_from_open_source_model"]

import typing as ty
from loguru import logger
import streamlit as st
import uuid
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.utils.interactive_env import is_interactive_env
from langchain_core.runnables.config import RunnableConfig
from langchain.agents import AgentExecutor
from openai import OpenAI

from svsvllm.utils import pop_params_not_in_fn
from .session_state import SessionState
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
    state = SessionState().state
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


def setup_for_streaming(
    pipeline_kwargs: dict[str, ty.Any] | None = None,
    use_react_agent: bool = None,
) -> tuple[CompiledGraph | AgentExecutor, RunnableConfig]:
    """Sets up all that is needed to have an agent ready to stream.

    Args:
        pipeline_kwargs (dict, optional):
            See :class:`HuggingFacePipeline`.

        use_react_agent (bool, optional):
            Whether to create an agent via `create_react_agent` (which gives you a `CompiledGraph` agent), or via `create_tool_calling_agent` then `AgentExecutor` (thus giving you a `AgentExecutor`).
            Defaults to `None`, meaning the value is taken from the state.

    Returns:
        agent_executor (CompiledGraph):
            Agent.

        agent_config (RunnableConfig):
            Agent's configuration.
    """
    # Initialize model
    logger.trace(f"Initializing model")
    create_chat_model(pipeline_kwargs=pipeline_kwargs)

    # Initialize RAG
    logger.trace(f"Initializing RAG and history-aware retriever")
    initialize_rag()
    create_history_aware_retriever()

    # Create agent
    logger.trace(f"Initializing agent")
    agent_executor = create_agent(use_react_agent=use_react_agent)

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
    agent_executor: CompiledGraph | AgentExecutor,
    agent_config: RunnableConfig,
    **kwargs: ty.Any,
) -> ty.Iterator[str | ty.Any]:
    """Stream response from agent.

    Args:
        query (str):
            Input query.

        agent_executor (CompiledGraph | AgentExecutor, optional):
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
    kwargs = pop_params_not_in_fn(agent_executor.stream, params=kwargs)
    logger.trace(f"Kwargs: {kwargs}")

    # NOTE: Stream yields events like: `{'alist': ['Ex for stream_mode="values"'], 'another_list': []}`
    for event in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]},
        config=agent_config,
        **kwargs,
    ):
        logger.trace(f"Event ({type(event)}): {event}")

        # If not a `dict`, yield it
        if not isinstance(event, dict):
            yield event

        # Get the messages
        messages: list[BaseMessage] = event.get("messages", None)
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


def get_response_from_open_source_model(
    query: str,
    pipeline_kwargs: dict[str, ty.Any] | None = None,
    agent_executor: CompiledGraph | AgentExecutor | None = None,
    agent_config: RunnableConfig | None = None,
    **kwargs: ty.Any,
) -> ty.Iterator[str | ty.Any]:
    """Stream response (given input query) from the open-source LLM.

    Args:
        query (str):
            Input query.

        pipeline_kwargs (dict, optional):
            See :class:`HuggingFacePipeline`.

        agent_executor (CompiledGraph | AgentExecutor, optional):
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
