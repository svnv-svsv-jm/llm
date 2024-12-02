__all__ = ["create_agent"]

from loguru import logger
import streamlit as st
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub
from langchain_core.runnables import Runnable

from .session_state import SessionState


def create_agent(use_react_agent: bool = None) -> CompiledGraph | AgentExecutor:
    """Create a new agent.

    Args:
        use_react_agent (bool, optional):
            Whether to create an agent via `create_react_agent` (which gives you a `CompiledGraph` agent), or via `create_tool_calling_agent` then `AgentExecutor` (thus giving you a `AgentExecutor`).
            Defaults to `None`, meaning the value is taken from the state.

    Returns:
        CompiledGraph | AgentExecutor: _description_
    """
    state = SessionState().state

    # Inputs
    if use_react_agent is None:
        use_react_agent = state.use_react_agent

    # Get `history_aware_retriever`
    logger.trace("Getting `history_aware_retriever`")
    history_aware_retriever = state.history_aware_retriever
    assert history_aware_retriever is not None, "Retriever not initialized."
    logger.trace(f"Got {history_aware_retriever}")

    # Create tools
    logger.trace("Creating tools")
    tool = create_retriever_tool(
        history_aware_retriever,  # type: ignore
        name="document_retriever",
        description="Searches and returns excerpts from the local database of documents.",
    )
    logger.trace(f"Created tool: {tool}")
    tools = [tool]

    # Memory management
    memory = MemorySaver()

    # Get LLM
    chat_model = state.chat_model
    assert chat_model is not None, "Chat model is not initialized."
    logger.trace(f"LLM: {chat_model}")

    # Create agent
    logger.trace("Creating agent")
    agent_executor: CompiledGraph | AgentExecutor
    if use_react_agent:
        logger.trace("Creating react agent")
        agent_executor = create_react_agent(chat_model, tools, checkpointer=memory)
    else:
        logger.trace("Creating tool-calling agent")
        prompt = hub.pull("hwchase17/openai-functions-agent")
        logger.trace(f"Prompt: '{prompt}'")
        agent: Runnable = create_tool_calling_agent(chat_model, tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)

    # Save and return
    logger.trace(f"Created agent: {agent_executor}")
    state.agent = agent_executor
    return agent_executor
