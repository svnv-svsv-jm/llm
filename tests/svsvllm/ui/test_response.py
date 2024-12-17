import pytest
from unittest.mock import patch, Mock, MagicMock
from loguru import logger
import typing as ty
import sys, os

import pprint
from codetiming import Timer as CommandTimer
from streamlit.testing.v1 import AppTest
import torch
from langgraph.graph.graph import CompiledGraph

from svsvllm.ui.response import setup_for_streaming, stream_open_source_model


@pytest.mark.parametrize("query", ["hi", "what day is today?"])
@pytest.mark.parametrize("use_mlx", [True])
def test_get_response_from_open_source_model(
    apptest_ss: AppTest,
    mock_rag_docs: str,
    device: torch.device,
    model_id: str,
    pipeline_kwargs: dict,
    query: str,
    use_mlx: bool,
    mlx_model_id: str,
) -> None:
    """Test `get_response_from_open_source_model`."""
    # Set up streaming
    model_name = mlx_model_id if use_mlx else model_id
    agent, cfg = setup_for_streaming(
        model_name=model_name,
        pipeline_kwargs=pipeline_kwargs,
        use_mlx=use_mlx,
    )

    logger.info(f"Config: {cfg}")
    logger.info(f"Agent: {agent}")

    assert isinstance(agent, CompiledGraph)

    # Log stuff for debugging
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

    # Early stop?
    if not use_mlx:
        logger.success("No mlx.")
        return

    # Stream response
    with CommandTimer("Streaming"):
        streamed_responses = []
        for r in stream_open_source_model(
            query,
            agent_executor=agent,
            agent_config=cfg,
            pipeline_kwargs=pipeline_kwargs,
            debug=True,
            stream_mode="debug",
        ):
            logger.info(f"streamed: {r}")
            streamed_responses.append(r)
    logger.success(streamed_responses)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
