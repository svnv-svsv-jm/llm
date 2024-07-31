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
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable, RunnablePassthrough
from transformers.utils import generic
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
from optimum.quanto import QuantizedModelForCausalLM, qint4

from svsvllm.scraping.driver import create_driver, DRIVER_TYPE


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
def directory_loader() -> DirectoryLoader:
    """`DirectoryLoader` object."""
    loader = DirectoryLoader(
        path=os.path.join("res", "documents"),
        glob="*.pdf",
        recursive=True,
    )
    logger.debug(f"Loader: {loader}")
    return loader


@pytest.fixture
def documents(directory_loader: DirectoryLoader) -> ty.List[Document]:
    """Loaded documents."""
    docs: ty.List[Document] = directory_loader.load()
    return docs


@pytest.fixture
def chunked_docs(documents: ty.List[Document]) -> ty.List[Document]:
    """Chunked documents."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
    chunked_docs = splitter.split_documents(documents)
    return chunked_docs


@pytest.fixture
def embeddings() -> HuggingFaceEmbeddings:
    """For all model names, see: https://www.sbert.net/docs/pretrained_models.html."""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


@pytest.fixture
def database(chunked_docs: ty.List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Database for the RAG."""
    logger.debug("Database for documents...")
    db = FAISS.from_documents(chunked_docs, embedding=embeddings)
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


def load_model(
    model_name: str,
    bnb_config: BitsAndBytesConfig | None = None,
) -> ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """Helper to load and quantize a model."""
    logger.debug(f"Loading model '{model_name}'...")
    try:
        model = QuantizedModelForCausalLM.from_pretrained(f"models/{model_name}")
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            token=os.environ["HUGGINGFACE_TOKEN"],
            revision="float16",
        )
        # Quantize with Optimum?
        if torch.backends.mps.is_available():
            model = QuantizedModelForCausalLM.quantize(model, weights=qint4, exclude="lm_head")
            model.save_pretrained(f"models/{model_name}")
    if isinstance(model, QuantizedModelForCausalLM):
        model = model._wrapped
    logger.debug(f"Loaded model '{model}'...")
    logger.debug(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HUGGINGFACE_TOKEN"])
    logger.debug(f"Loaded tokenizer '{tokenizer}'...")
    return model, tokenizer


@pytest.fixture
def cerbero(
    bnb_config: BitsAndBytesConfig | None,
) -> ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer]:
    """Cerbero."""
    model_name = "galatolo/cerbero-7b"  # Italian
    model, tokenizer = load_model(model_name, bnb_config=bnb_config)
    return model, tokenizer


@pytest.fixture
def cerbero_pipeline(
    cerbero: ty.Tuple[AutoModelForCausalLM | QuantizedModelForCausalLM, AutoTokenizer],
    device: torch.device,
) -> HuggingFacePipeline:
    """Cerbero pipeline."""
    logger.debug("Pipeline...")
    model, tokenizer = cerbero
    # with patch.object(QuantizedModelForCausalLM, "__module__", return_value="torch"):
    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
        device=device,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    logger.debug(f"Pipeline: {llm}")
    return llm


@pytest.fixture
def italian_prompt_template() -> str:
    """Italian prompt template for the LLM."""
    return """
<|system|>
Rispondi alla domanda con la tua conoscenza. Usa il seguente contesto per aiutarti:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>
"""


@pytest.fixture
def llm_chain_cerbero(
    cerbero_pipeline: HuggingFacePipeline,
    italian_prompt_template: str,
) -> RunnableSerializable:
    """LLM chain for Cerbero."""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=italian_prompt_template,
    )
    llm_chain = prompt | cerbero_pipeline | StrOutputParser()
    logger.debug(f"LLM chain: {llm_chain}")
    return llm_chain


@pytest.fixture
def llm_chain_cerbero_w_rag(
    llm_chain_cerbero: RunnableSerializable,
    database: FAISS,
) -> RunnableSerializable:
    """LLM chain for Cerbero."""
    retriever = database.as_retriever()
    rag_chain: RunnableSerializable = {
        "context": retriever,
        "question": RunnablePassthrough(),
    } | llm_chain_cerbero
    logger.debug(f"LLM+RAG chain: {rag_chain}")
    return rag_chain
