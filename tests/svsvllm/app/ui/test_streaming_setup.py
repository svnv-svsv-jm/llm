import pytest
from unittest.mock import patch, Mock, MagicMock
from loguru import logger
import typing as ty
import sys, os

import pprint
import streamlit as st
from streamlit.testing.v1 import AppTest
import torch
from langgraph.graph.graph import CompiledGraph
from langchain.agents import AgentExecutor
from langchain_core.runnables.base import RunnableBinding
from langchain_huggingface import ChatHuggingFace

from svsvllm.ui.response import setup_for_streaming


@pytest.mark.parametrize("query", ["hi"])
@pytest.mark.parametrize("use_react_agent", [False, True])
def test_streaming_setup(
    apptest_ss: AppTest,
    mock_rag_docs: str,
    device: torch.device,
    pipeline_kwargs: dict,
    query: str,
    use_react_agent: bool,
) -> None:
    """Test setup for streaming is correct."""
    # Set up streaming
    agent, cfg = setup_for_streaming(pipeline_kwargs=pipeline_kwargs, use_react_agent=use_react_agent)
    logger.info(f"Config: {cfg}")
    logger.info(f"Agent: {agent}")

    # Test correct class depending on `use_react_agent`
    if not use_react_agent:
        assert isinstance(agent, AgentExecutor)
    else:
        assert isinstance(agent, CompiledGraph)

    # Based on the class, we log and test different things
    if isinstance(agent, CompiledGraph):
        logger.info(f"agent.config: {agent.config}")
        # Attempt to list tools or nodes in the graph
        tools = agent.nodes  # or `agent.graph.nodes` if it's nested
        # Info on tools
        for name, node in tools.items():
            logger.info(f"{name}: {[(key, type(item)) for key, item in node.__dict__.items()]}")
            bound = node.bound
            logger.info(f"Bound: {[(key, type(item)) for key, item in bound.__dict__.items()]}")
            tools_by_name: dict = bound.__dict__.get("tools_by_name", {})
            logger.info(
                f"Tools: {[(key, type(item)) for key, item in tools_by_name.items()]}\n{pprint.pformat(tools_by_name, indent=2)}"
            )
    else:
        # Let's log all we can and find our LLM, so that we can test it has the right `pipeline_kwargs`
        for key, item in agent.agent.__dict__.items():
            logger.info(f"[{type(agent)}.{type(agent.agent)}] {key}: {type(item)}")
        # Unravel the runnable
        runnable = agent.agent.__dict__.get("runnable", None)
        logger.info(f"Runnable ({type(runnable)}):\n{runnable}")
        if runnable is not None:
            for key, item in runnable.__dict__.items():
                logger.info(f"[Runnable] {key}: {type(item)}")
                if isinstance(item, list):
                    for i, v in enumerate(item):
                        logger.info(f"[Runnable:{key}[{i}]] ({type(v)}): \n{v}\n\n")
                        if isinstance(v, RunnableBinding):
                            # Here there should be our LLM
                            if isinstance(v.bound, ChatHuggingFace):
                                logger.info(f"{v.bound}")
                                chat_model = v.bound
                                llm = chat_model.llm
                                assert (
                                    llm.pipeline_kwargs == pipeline_kwargs
                                ), f"Expected {pipeline_kwargs} but got {llm.pipeline_kwargs}"

    # with CommandTimer(f"{agent.__class__.__name__}.invoke"):
    #     out = agent.invoke({"input": [HumanMessage(content=query)]}, config=cfg)
    # logger.info(f"Response: {out}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
