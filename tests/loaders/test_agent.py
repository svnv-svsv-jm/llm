import pytest
from loguru import logger
import typing as ty
import sys, os

from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models import ChatHuggingFace
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableSerializable
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from svsvllm.loaders import load_agent


# TODO: see https://huggingface.co/blog/open-source-llms-as-agents
def test_load_documents_error(
    retriever: VectorStoreRetriever,
) -> None:
    """Test `load_agent` and check it raises an error."""
    llm = HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama_v1.1", model="TinyLlama/TinyLlama_v1.1")
    chat_model = ChatHuggingFace(llm=llm)

    # Create agent
    agent = load_agent(
        llm=llm,
        retriever=retriever,
        retriever_name="retriever",
        retriever_description="description",
    )

    # Test
    query = "What is Task Decomposition?"
    for event in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
