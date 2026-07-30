import os
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import glob

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
        os.path.join(docs_path, "**", "*.pdf"),
        recursive=True
    )

    for pdf_file in pdf_files:

        print("Loading", pdf_file)

        pdf_loader = PyPDFLoader(pdf_file)
        pdf_documents.extend(pdf_loader.load())

    txt_documents = text_loader.load()

    documents = txt_documents + pdf_documents
    

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No supported documents (.txt or .pdf) found in '{docs_path}'."
        )

    print(f"Loaded {len(documents)} document(s).\n")

    for i, doc in enumerate(documents, start=1):
        print(f"Document {i}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")

        if "page" in doc.metadata:
            print(f"Page: {doc.metadata['page']}")

        print()
    
    return documents


def split_documents(documents, chunk_size=900, chunk_overlap=150):
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n"],
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5], start=1):

            print(f"\n---- Chunk {i} ----")
            print(f"Source: {chunk.metadata.get('source', 'Unknown')}")

            if "page" in chunk.metadata:
                print(f"Page: {chunk.metadata['page']}")

            print(f"Length: {len(chunk.page_content)} characters")
            print("Content:")
            print(chunk.page_content)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(
        model="bge-m3"
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    batch_size = 200

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i + batch_size]

        print(
            f"Adding batch {i // batch_size + 1} "
            f"({len(batch)} documents)"
        )

        vectorstore.add_documents(batch)

    print("Finished creating vector store")
    print(f"Total documents stored: {vectorstore._collection.count()}")

    return vectorstore


def main():

    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    if os.path.exists(persistent_directory):

        print("Vector store already exists. No need to re-process documents.")

        embedding_model = OllamaEmbeddings(
            model="bge-m3"
        )

        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

        print(
            f"Loaded existing vector store with "
            f"{vectorstore._collection.count()} documents"
        )

        return vectorstore

    print("Initializing vector store...\n")

    # Step 1: Load documents
    documents = load_documents(docs_path)

    # Step 2: Split documents
    chunks = split_documents(documents)

    # Step 3: Create vector store
    vectorstore = create_vector_store(
        chunks,
        persistent_directory
    )

    print("\nIngestion complete! Your documents are now ready for RAG queries.")

    return vectorstore


if __name__ == "__main__":
    main()