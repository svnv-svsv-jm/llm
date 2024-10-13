__all__ = ["get_openai_response", "get_response_from_open_source_model"]

import typing as ty
import streamlit as st
from openai import OpenAI
import uuid
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.utils.interactive_env import is_interactive_env

from .model import create_chat_model
from .rag import initialize_rag, create_retriever
from .agent import create_agent


def get_openai_response(openai_api_key: str, model: str) -> str:
    """Returns the response from the OpenAI model.
    The full chat history up to now is fed as input.
    """
    # OpenAI
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=st.session_state.messages,
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
    # Initialize model
    create_chat_model()

    # Initialize RAG
    initialize_rag()
    create_retriever()

    # Create agent
    create_agent()

    # Create chat configuration
    thread_id = st.session_state.get("thread_id", None)
    if thread_id is None:
        thread_id = uuid.uuid4()
        st.session_state["thread_id"] = thread_id
    agent_config = {"configurable": {"thread_id": thread_id}}
    st.session_state["agent_config"] = agent_config

    agent_executor: CompiledGraph = st.session_state.agent

    # Return stream
    # Stream yields events like: `{'alist': ['Ex for stream_mode="values"'], 'another_list': []}`
    # return agent_executor.stream(  # type: ignore
    #     {"messages": [HumanMessage(content=query)]},
    #     config=st.session_state.agent_config,
    #     stream_mode="values",
    # )
    for event in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]},
        config=st.session_state.agent_config,
        stream_mode="values",
    ):
        last_message: BaseMessage = event["messages"][-1]
        yield last_message.pretty_repr(html=is_interactive_env())
