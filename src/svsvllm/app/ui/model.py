import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
import torch

from svsvllm.loaders import load_model
from svsvllm.utils import find_device


def create_chat_model() -> None:
    """Create a chat model."""
    hf_model, tokenizer = load_model(st.session_state.model_name)
    st.session_state["hf_model"] = hf_model
    st.session_state["tokenizer"] = tokenizer
    llm = HuggingFacePipeline(
        pipeline=pipeline(
            task="text-generation",
            model=hf_model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map=find_device(),
        )
    )
    chat_model = ChatHuggingFace(llm=llm)
    st.session_state["chat_model"] = chat_model
