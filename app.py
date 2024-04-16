import argparse
import sys
import os

from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from svsvllm.models import check_if_model_is_available
from svsvllm.loaders import load_documents


TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


PROMPT_TEMPLATE = """
### Instruction:
You're helpful assistant, who answers questions based upon provided research in a distinct and clear way.

## Research:
{context}

## Question:
{question}
"""


PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


def load_documents_into_database(model_name: str, documents_path: str) -> Chroma:
    """
    Loads documents from the specified directory into the Chroma database
    after splitting the text into chunks.

    Returns:
        Chroma: The Chroma database with loaded documents.
    """

    print("Loading documents")
    raw_documents = load_documents(documents_path)
    documents = TEXT_SPLITTER.split_documents(raw_documents)

    print("Creating embeddings and loading documents into Chroma")
    db = Chroma.from_documents(
        documents,
        OllamaEmbeddings(model=model_name),
    )
    return db


def main(llm_model_name: str, embedding_model_name: str, documents_path: str) -> None:
    """Main entrypoint."""
    # Check to see if the models available, if not attempt to pull them
    check_if_model_is_available(llm_model_name)
    check_if_model_is_available(embedding_model_name)

    # Creating database form documents
    try:
        db = load_documents_into_database(embedding_model_name, documents_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    llm = Ollama(
        model=llm_model_name,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(search_kwargs={"k": 8}),
        chain_type_kwargs={"prompt": PROMPT},
    )

    while True:
        try:
            user_input = input("\n\nPlease enter your question (or type 'exit' to end): ")
            if user_input.lower() == "exit":
                break

            # docs = db.similarity_search(user_input)
            qa_chain.invoke({"query": user_input})
        except KeyboardInterrupt:
            break
    print("Exiting...")


def parse_arguments() -> argparse.Namespace:
    """Parser."""
    parser = argparse.ArgumentParser(description="Run local LLM with RAG with Ollama.")
    parser.add_argument(
        "-m",
        "--model",
        default="mistral",
        help="The name of the LLM model to use.",
    )
    parser.add_argument(
        "-e",
        "--embedding_model",
        default="nomic-embed-text",
        help="The name of the embedding model to use.",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=os.path.join("res", "documents"),
        help="The path to the directory containing documents to load.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.model, args.embedding_model, args.path)
