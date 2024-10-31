__all__ = [
    "DEFAULT_LLM",
    "OPENAI_DEFAULT_MODEL",
    "EMBEDDING_DEFAULT_MODEL",
    "ZEPHYR_CHAT_TEMPLATE",
    "CUSTOM_CHAT_TEMPLATE",
]

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

DEFAULT_LLM = "TinyLlama/TinyLlama_v1.1"
OPENAI_DEFAULT_MODEL = "gpt-3.5-turbo"
EMBEDDING_DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

# Copied from: HuggingFaceH4/zephyr-7b-beta
ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

# Custom
CUSTOM_CHAT_TEMPLATE = [
    SystemMessage(content="You are a helpful assistant.").model_dump(),
    AIMessage(content="How can I help you today?").model_dump(),
    HumanMessage(content="I'd like to show off how chat templating works!").model_dump(),
]
