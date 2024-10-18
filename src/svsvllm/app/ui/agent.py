import streamlit as st
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool


def create_agent() -> None:
    """Create a new agent."""
    tool = create_retriever_tool(
        st.session_state.history_aware_retriever,
        name="document_retriever",
        description="Searches and returns excerpts from the local database of documents.",
    )
    tools = [tool]
    memory = MemorySaver()
    agent_executor = create_react_agent(st.session_state.chat_model, tools, checkpointer=memory)
    st.session_state["agent"] = agent_executor