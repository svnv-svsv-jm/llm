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
    agent, cfg = setup_for_streaming(pipeline_kwargs=pipeline_kwargs)
    logger.info(f"Config: {cfg}")
    logger.info(f"Agent: {agent}")

    # Based on the class, we log and test different things
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

    # with CommandTimer(f"{agent.__class__.__name__}.invoke"):
    #     out = agent.invoke({"input": [HumanMessage(content=query)]}, config=cfg)
    # logger.info(f"Response: {out}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
