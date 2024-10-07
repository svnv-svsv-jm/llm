import pytest
from loguru import logger
import typing as ty
import sys, os

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


def test_agent_from_original_docs(
    retriever: VectorStoreRetriever,
    tiny_llama_pipeline: RunnableSerializable,
) -> None:
    """Test based on https://python.langchain.com/docs/tutorials/qa_chat_history/#tying-it-together-1"""
    # Create LLM
    # Original tutorial does: `llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)`, but we do not have a API key.
    llm = ChatHuggingFace(
        llm=HuggingFacePipeline(
            verbose=True,
            pipeline=tiny_llama_pipeline,
        )
    )

    # Build retriever tool
    tool = create_retriever_tool(
        retriever,
        name="document_retriever",
        description="Searches and returns excerpts from the local database of documents.",
    )
    tools = [tool]

    # Create agent
    memory = MemorySaver()
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)

    # Create query
    query = "Deep Reinforcement Learning (Deep RL) is increasingly used to cope with the open-world assumption in service-oriented systems. Is this true?"

    # Invoke
    config = RunnableConfig(configurable={"thread_id": "abc123"})
    for event in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
