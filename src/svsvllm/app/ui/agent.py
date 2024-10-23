__all__ = ["create_agent"]

import streamlit as st
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool

from .session_state import SessionState


def create_agent() -> None:
    """Create a new agent."""
    state = SessionState().state
    tool = create_retriever_tool(
        state.history_aware_retriever,
        name="document_retriever",
        description="Searches and returns excerpts from the local database of documents.",
    )
    tools = [tool]
    memory = MemorySaver()
    agent_executor = create_react_agent(state.chat_model, tools, checkpointer=memory)
    state.agent = agent_executor
