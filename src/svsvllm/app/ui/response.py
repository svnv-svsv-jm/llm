__all__ = ["get_openai_response", "get_response_from_open_source_model"]

import typing as ty
from loguru import logger
import streamlit as st
import uuid
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.utils.interactive_env import is_interactive_env
from langchain_core.runnables.config import RunnableConfig
from openai import OpenAI

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


def get_response_from_open_source_model(
    query: str,
) -> ty.Iterator[str | dict[str, list[BaseMessage | ty.Any]]]:
    """Work in progress."""
    logger.trace(f"Query: {query}")

    # Initialize model
    logger.trace(f"Initializing model")
    create_chat_model()

    # Initialize RAG
    logger.trace(f"Initializing RAG and history-aware retriever")
    initialize_rag()
    create_history_aware_retriever()

    # Create agent
    logger.trace(f"Initializing agent")
    create_agent()

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

    # Retrieve agent: must be initialized
    agent_executor: CompiledGraph = st.session_state.agent

    # Return stream
    # NOTE: Stream yields events like: `{'alist': ['Ex for stream_mode="values"'], 'another_list': []}`
    logger.trace(f"Streaming")

    # return agent_executor.stream(  # type: ignore
    #     {"messages": [HumanMessage(content=query)]},
    #     config=st.session_state.agent_config,
    #     stream_mode="values",
    # )
    for event in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]},
        config=agent_config,
        stream_mode="values",
    ):
        last_message: BaseMessage = event["messages"][-1]
        yield last_message.pretty_repr(html=is_interactive_env())
