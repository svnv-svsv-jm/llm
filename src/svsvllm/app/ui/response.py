__all__ = ["get_response"]

import typing as ty
import streamlit as st
from openai import OpenAI
import uuid
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import HumanMessage

from .const import OPEN_SOURCE_MODELS_SUPPORTED
from .defaults import DEFAULT_MODEL
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
) -> str | ty.Iterator[dict[str, ty.Any] | ty.Any]:
    """Work in progress."""
    # TODO: initialize RAG, initialize model, create tools, create agent.
    if OPEN_SOURCE_MODELS_SUPPORTED:
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
        return agent_executor.stream(
            {"messages": [HumanMessage(content=query)]},
            config=st.session_state.agent_config,
            stream_mode="values",
        )

    # Let the chatbox inform the user
    return "Welcome to FiscalAI! Unfortunately, support for open-source models is still in development. Please add your OpenAI API key to get a different, meaningful response."


def get_response(
    model: str = DEFAULT_MODEL,
    openai_api_key: str | None = None,
) -> str:
    """Get response from the chatbot.

    Args:
        model (str | None, optional):
            Model ID or name.
            Defaults to `"gpt-3.5-turbo"`.

        openai_api_key (str | None, optional):
            OpenAI key.
            Defaults to `None`.

    Returns:
        str: response from the LLM.
    """
    # No OpenAI
    if not openai_api_key:
        return get_response_from_open_source_model()

    # OpenAI
    msg = get_openai_response(openai_api_key, model)

    # Sanity check and return
    if msg is None:
        msg = ""
    return f"{msg}"
