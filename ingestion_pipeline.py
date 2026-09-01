import os
import glob
import time
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ======================================================
# Configuration
# ======================================================

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")

JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"
EMBEDDING_MODEL = "jina-embeddings-v5-text-small"

PERSIST_DIRECTORY = "db/chroma_db"


# ======================================================
# Jina Embeddings
# ======================================================

class JinaEmbeddings:

    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model

    def _embed(self, texts, task):

        max_retries = 6

        for attempt in range(max_retries):

            try:

                response = requests.post(
                    JINA_EMBEDDING_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "input": texts,
                        "task": task,
                        "dimensions": 1024,
                        "normalized": True,
                        "embedding_type": "float"
                    },
                    timeout=180
                )

                if response.status_code == 429:

                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        wait_time = min(
                            10 * (2 ** attempt),
                            120
                        )

                    print(
                        f"Jina rate limit reached. "
                        f"Waiting {wait_time} seconds..."
                    )

                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                data = response.json()["data"]

                data.sort(
                    key=lambda item: item["index"]
                )

                return [
                    item["embedding"]
                    for item in data
                ]

            except requests.exceptions.Timeout:

                wait_time = min(
                    10 * (2 ** attempt),
                    120
                )

                print(
                    f"Jina request timed out. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            except requests.exceptions.ConnectionError:

                wait_time = min(
                    10 * (2 ** attempt),
                    120
                )

                print(
                    f"Connection error. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

        raise RuntimeError(
            "Jina embedding request failed "
            f"after {max_retries} attempts."
        )

    def embed_documents(self, texts):

        return self._embed(
            texts,
            "retrieval.passage"
        )

    def embed_query(self, text):

        return self._embed(
            [text],
            "retrieval.query"
        )


# ======================================================
# Load Documents
# ======================================================

def load_documents(docs_path):

    print("Loading documents from", docs_path)

    if not os.path.exists(docs_path):

        raise FileNotFoundError(
            f"The directory '{docs_path}' does not exist."
        )

    # Load TXT files
    text_loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8",
            "autodetect_encoding": True
        }
    )

    # Load PDF files
    pdf_documents = []

    pdf_files = glob.glob(
        os.path.join(
            docs_path,
            "**",
            "*.pdf"
        ),
        recursive=True
    )

    for pdf_file in pdf_files:

        print("Loading", pdf_file)

        pdf_loader = PyPDFLoader(
            pdf_file
        )

        pdf_documents.extend(
            pdf_loader.load()
        )

    txt_documents = text_loader.load()

    documents = (
        txt_documents +
        pdf_documents
    )

    if len(documents) == 0:

        raise FileNotFoundError(
            f"No supported documents (.txt or .pdf) "
            f"found in '{docs_path}'."
        )

    print(
        f"Loaded {len(documents)} document(s).\n"
    )

    for i, doc in enumerate(
        documents,
        start=1
    ):

        print(f"Document {i}")

        print(
            f"Source: "
            f"{doc.metadata.get('source', 'Unknown')}"
        )

        if "page" in doc.metadata:

            print(
                f"Page: "
                f"{doc.metadata['page']}"
            )

        print()

    return documents


# ======================================================
# Split Documents
# ======================================================

def split_documents(
    documents,
    chunk_size=900,
    chunk_overlap=150
):

    print(
        "Splitting documents into chunks..."
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n"
        ],
    )

    chunks = text_splitter.split_documents(
        documents
    )

    if chunks:

        for i, chunk in enumerate(
            chunks[:5],
            start=1
        ):

            print(
                f"\n---- Chunk {i} ----"
            )

            print(
                f"Source: "
                f"{chunk.metadata.get('source', 'Unknown')}"
            )

            if "page" in chunk.metadata:

                print(
                    f"Page: "
                    f"{chunk.metadata['page']}"
                )

            print(
                f"Length: "
                f"{len(chunk.page_content)} "
                f"characters"
            )

            print("Content:")

            print(
                chunk.page_content
            )

        if len(chunks) > 5:

            print(
                f"\n... and "
                f"{len(chunks) - 5} more chunks"
            )

    return chunks


# ======================================================
# Create Vector Store
# ======================================================

def create_vector_store(
    chunks,
    persist_directory=PERSIST_DIRECTORY
):

    print(
        "Creating embeddings and storing "
        "in ChromaDB..."
    )

    if not JINA_API_KEY:

        raise ValueError(
            "JINA_API_KEY is not set. "
            "Add it to your .env file."
        )

    embedding_model = JinaEmbeddings(
        api_key=JINA_API_KEY,
        model=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

    # Send 100 chunks per request
    batch_size = 100

    total_batches = (
        len(chunks) + batch_size - 1
    ) // batch_size

    for i in range(
        0,
        len(chunks),
        batch_size
    ):

        batch = chunks[
            i:i + batch_size
        ]

        batch_number = (
            i // batch_size
        ) + 1

        print(
            f"\nEmbedding batch "
            f"{batch_number}/{total_batches} "
            f"({len(batch)} chunks)"
        )

        vectorstore.add_documents(
            batch
        )

        print(
            f"Batch {batch_number} completed."
        )

        # Small delay between requests
        if batch_number < total_batches:

            time.sleep(2)

    print(
        "\nFinished creating vector store."
    )

    print(
        f"Total documents stored: "
        f"{vectorstore._collection.count()}"
    )

    return vectorstore


# ======================================================
# Main
# ======================================================

def main():

    docs_path = "docs"

    persistent_directory = (
        PERSIST_DIRECTORY
    )

    if os.path.exists(
        persistent_directory
    ):

        print(
            "Vector store already exists. "
            "No need to re-process documents."
        )

        if not JINA_API_KEY:

            raise ValueError(
                "JINA_API_KEY is not set. "
                "Add it to your .env file."
            )

        embedding_model = JinaEmbeddings(
            api_key=JINA_API_KEY,
            model=EMBEDDING_MODEL
        )

        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={
                "hnsw:space": "cosine"
            },
        )

        print(
            f"Loaded existing vector store "
            f"with "
            f"{vectorstore._collection.count()} "
            f"documents"
        )

        return vectorstore

    print(
        "Initializing vector store...\n"
    )

    # Step 1: Load documents
    documents = load_documents(
        docs_path
    )

    # Step 2: Split documents
    chunks = split_documents(
        documents
    )

    # Step 3: Create vector store
    vectorstore = create_vector_store(
        chunks,
        persistent_directory
    )

    print(
        "\nIngestion complete! "
        "Your documents are now ready "
        "for RAG queries."
    )

    return vectorstore


if __name__ == "__main__":
    main()