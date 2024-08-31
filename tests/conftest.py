import pytest
from unittest.mock import patch
import sys, os, pyrootutils
import typing as ty
from loguru import logger
import time

pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)

import torch
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer,
    MixtralForCausalLM,
)
from optimum.quanto import QuantizedModelForCausalLM

from svsvllm.scraping.driver import create_driver, DRIVER_TYPE
from svsvllm.loaders import load_model, load_documents
from svsvllm.rag import create_rag_database


@pytest.fixture
def artifact_location() -> str:
    """Location for test artifacts."""
    return "pytest_artifacts"


@pytest.fixture
def web_driver() -> ty.Generator[DRIVER_TYPE, None, None]:
    """Web driver."""
    # Headless only if running all tests
    headless = False  # __file__ not in sys.argv[0]
    # Create driver
    driver = create_driver(
        "window-size=1200x600",
        headless=headless,
        use_firefox=False,
    )
    yield driver
    # Sleep
    time.sleep(1)
    driver.quit()


@pytest.fixture
def device() -> torch.device:
    """Torch device."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    logger.debug(f"Device: {device}")
    return device


@pytest.fixture
def docs_path() -> str:
    """Path to foler with documents."""
    path = os.path.join("res", "documents")
    return path


@pytest.fixture
def documents(docs_path: str) -> ty.List[Document]:
    """Loaded documents."""
    docs: ty.List[Document] = load_documents(docs_path)
    return docs


@pytest.fixture
def database(docs_path: str) -> FAISS:
    """Database for the RAG."""
    logger.debug("Database for documents...")
    db = create_rag_database(docs_path)
    return db


@pytest.fixture
def retriever(database: FAISS) -> VectorStoreRetriever:
    """`VectorStoreRetriever` object."""
    logger.debug("Retriever...")
    retriever = database.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever


@pytest.fixture
def bnb_config() -> BitsAndBytesConfig | None:
    """Quantization configuration with BitsAndBytes."""
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        bnb_config = None
    logger.debug(f"Config: {bnb_config}")
    return bnb_config


@pytest.fixture
def cerbero(
    bnb_config: BitsAndBytesConfig | None,
) -> ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """Cerbero."""
    model_name = "galatolo/cerbero-7b"  # Italian
    model, tokenizer = load_model(
        model_name,
        bnb_config=bnb_config,
        quantize=True,
        quantize_w_torch=False,
    )
    return model, tokenizer


@pytest.fixture
def mistral_small(
    bnb_config: BitsAndBytesConfig | None,
) -> ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """Small mistral."""
    model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    model, tokenizer = load_model(
        model_name,
        bnb_config=bnb_config,
        quantize=True,
        quantize_w_torch=False,
        model_class=MixtralForCausalLM,
        tokenizer_class=LlamaTokenizer,
    )
    return model, tokenizer
