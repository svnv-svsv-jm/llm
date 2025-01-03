__all__ = ["create_agent"]

from loguru import logger
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool

from svsvllm.exceptions import NoChatModelError
from .session_state import session_state


def create_agent() -> CompiledGraph:
    """Create a new agent.

    Returns:
        CompiledGraph: LLM agent.
    """
    # Get `history_aware_retriever`
    logger.trace("Getting `history_aware_retriever`")
    har = session_state.history_aware_retriever
    logger.trace(f"Got {har}")

    # Create tools
    tools = []
    logger.trace("Creating tools")
    if har is not None:
        tool = create_retriever_tool(
            har,  # type: ignore
            name="document_retriever",
            description="Searches and returns excerpts from the local database of documents.",
        )
        logger.trace(f"Created tool: {tool}")
        tools.append(tool)

    # Memory management
    memory = MemorySaver()

    # Get LLM
    chat_model = session_state.chat_model
    if chat_model is None:
        raise NoChatModelError("Chat model is not initialized.")

    # Create agent
    logger.trace(f"Creating react agent from: {type(chat_model)}")
    agent_executor = create_react_agent(chat_model, tools, checkpointer=memory)

    # Save and return
    logger.trace(f"Created agent: {agent_executor}")
    session_state.agent = agent_executor
    return agent_executor
