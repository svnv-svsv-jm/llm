__all__ = ["create_agent"]

from loguru import logger
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool

from .session_state import SessionState


def create_agent() -> CompiledGraph:
    """Create a new agent.

    Returns:
        CompiledGraph: LLM agent.
    """
    state = SessionState().state

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
    agent_executor: CompiledGraph
    logger.trace("Creating react agent")
    agent_executor = create_react_agent(chat_model, tools, checkpointer=memory)

    # Save and return
    logger.trace(f"Created agent: {agent_executor}")
    state.agent = agent_executor
    return agent_executor
