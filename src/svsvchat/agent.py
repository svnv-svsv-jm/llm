__all__ = ["create_agent"]

from loguru import logger
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue

from svsvllm.exceptions import NoChatModelError
from svsvllm.types import StateModifier
from .session_state import session_state
from .settings import settings


def create_agent(
    system_prompt: str = None,
    state_modifier: StateModifier | None = None,
    prompt_role: str | None = None,
) -> CompiledGraph:
    """Create a new agent.

    Args:
        system_prompt (str):
            System prompt.
            This message is appended BEFORE the user message.

        state_modifier (StateModifier):
            See :func:`~langgraph.prebuilt.create_react_agent`.
            This can be used to pass a prompt.
            For example, if an object of type `SystemMessage` is passed,
            this is added to the beginning of the list of messages in `state["messages"]`.

        prompt_role (str):
            The role for the system prompt.
            Normally, this is `"system"`, but some models do not support the `"system"` role.

    Raises:
        NoChatModelError:
            If the base LLM model is not initialized.

    Returns:
        CompiledGraph: LLM agent.
    """
    # Inputs
    if prompt_role is None:
        prompt_role = settings.prompt_role
    if system_prompt is None:
        system_prompt = settings.system_prompt
    if state_modifier is None:
        state_modifier = settings.system_prompt

    # Get `history_aware_retriever`
    logger.trace("Getting `history_aware_retriever`")
    har = session_state.history_aware_retriever
    logger.trace(f"Got {type(har)}")

    # Create tools
    tools = []
    logger.trace("Creating tools")
    if har is not None:
        tool = create_retriever_tool(
            har,  # type: ignore
            name="document_retriever",
            description="Searches and returns excerpts from the local database of documents.",
        )
        logger.trace(f"Created tool: {type(tool)}")
        tools.append(tool)

    # Memory management
    memory = MemorySaver()

    # Get LLM
    chat_model = session_state.chat_model
    if chat_model is None:
        raise NoChatModelError("Chat model is not initialized.")

    # Create agent
    logger.trace(f"Creating react agent from: {type(chat_model)}")

    # This section is inspired by: https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent
    if system_prompt:
        prompt = ChatPromptTemplate.from_messages(
            [
                (prompt_role, system_prompt),
                ("placeholder", "{messages}"),
            ]
        )

        def format_for_model(state: dict) -> PromptValue:
            """You can do more complex modifications here."""
            logger.trace(f"state: {state}")
            messages = state["messages"]
            logger.trace(f"messages: {messages}")
            prompt_value = prompt.invoke({"messages": messages})
            logger.trace(f"prompt_value ({type(prompt_value)}): {prompt_value}")
            messages = [f"{msg.content}" for msg in prompt_value.to_messages()]
            prompt_out = "\n".join(messages)
            logger.trace(f"prompt_out ({type(prompt_out)}): {prompt_out}")
            return prompt_out

        state_modifier = format_for_model

    agent_executor = create_react_agent(
        chat_model,
        tools,
        checkpointer=memory,
        state_modifier=state_modifier,
    )

    # Save and return
    logger.trace(f"Created agent: {agent_executor}")
    session_state.agent = agent_executor
    return agent_executor
