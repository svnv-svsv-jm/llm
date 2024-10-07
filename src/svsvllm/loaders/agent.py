__all__ = ["load_agent"]

from langchain.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph


def load_agent(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    retriever_name: str,
    retriever_description: str,
) -> CompiledGraph:
    """_summary_

    Args:
        llm (BaseChatModel): _description_
        retriever (VectorStoreRetriever): _description_
        retriever_name (str): _description_
        retriever_description (str): _description_

    Returns:
        CompiledGraph: _description_
    """
    memory = MemorySaver()
    tool = create_retriever_tool(retriever, name=retriever_name, description=retriever_description)
    tools = [tool]
    agent_executor = create_react_agent(
        llm,
        tools,
        checkpointer=memory,
    )
    return agent_executor
