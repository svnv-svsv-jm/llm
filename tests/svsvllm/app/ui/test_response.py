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
from langchain_core.messages import HumanMessage

from svsvllm.utils import CommandTimer
from svsvllm.ui.response import setup_for_streaming, get_response_from_open_source_model


@pytest.mark.parametrize("query", ["hi"])
@pytest.mark.parametrize("use_react_agent", [False, True])
def test_get_response_from_open_source_model(
    apptest_ss: AppTest,
    mock_rag_docs: str,
    device: torch.device,
    pipeline_kwargs: dict,
    query: str,
    use_react_agent: bool,
) -> None:
    """Test `get_response_from_open_source_model`."""
    # Set up streaming
    agent, cfg = setup_for_streaming(pipeline_kwargs=pipeline_kwargs, use_react_agent=use_react_agent)

    logger.info(f"Config: {cfg}")
    logger.info(f"Agent: {agent}")

    if not use_react_agent:
        assert isinstance(agent, AgentExecutor)
    else:
        assert isinstance(agent, CompiledGraph)

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
        for key, item in agent.agent.__dict__.items():
            logger.info(f"[{type(agent)}.{type(agent.agent)}] {key}: {type(item)}")
        runnable = agent.agent.__dict__.get("runnable", None)
        logger.info(f"Runnable ({type(runnable)}):\n{runnable}")
        if runnable is not None:
            for key, item in runnable.__dict__.items():
                logger.info(f"[Runnable] {key}: {type(item)}")

    with CommandTimer(f"{agent.__class__.__name__}.invoke"):
        out = agent.invoke({"input": [HumanMessage(content=query)]}, config=cfg)
    logger.info(f"Response: {out}")

    # Stream response
    with CommandTimer("Streaming"):
        for r in get_response_from_open_source_model(
            query,
            agent_executor=agent,
            agent_config=cfg,
            pipeline_kwargs=pipeline_kwargs,
            debug=True,
            stream_mode="debug",
        ):
            logger.info(f"streamed: {r}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s"])
