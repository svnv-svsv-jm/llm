import pytest
from loguru import logger
import typing as ty
import sys, os

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableSerializable, RunnableConfig


# TODO: see https://huggingface.co/blog/open-source-llms-as-agents
def test_agent_hack(
    retriever: VectorStoreRetriever,
    tiny_llama_pipeline: RunnableSerializable,
) -> None:
    """Test `load_agent`."""
    # Create LLM
    # Original tutorial does: `llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)`, but we do not have a API key.
    chat_model = ChatHuggingFace(
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

    # Setup ReAct style prompt
    prompt = hub.pull("hwchase17/react-json")
    prompt = prompt.partial(
        tools=render_text_description(tools),  # type: ignore
        tool_names=", ".join([t.name for t in tools]),
    )
    logger.info(f"Prompt: {prompt}")

    # Define the agent
    chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | chat_model_with_stop
        | ReActJsonSingleInputOutputParser()
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        # memory=MemorySaver(),
    )

    # Query
    query = "Deep Reinforcement Learning (Deep RL) is increasingly used to cope with the open-world assumption in service-oriented systems. Is this true?"

    # Invoke
    config = RunnableConfig(configurable={"thread_id": "abc123"})
    out = agent_executor.invoke({"input": query}, config=config)
    logger.info(out)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
