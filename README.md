# RAG Legal AI Assistant

## Overview

This repository implements a Retrieval-Augmented Generation (RAG) project for legal document question answering. It ingests text and PDF legal documents, creates a vector store using embeddings, and exposes a Streamlit interface for asking legal questions. The system is designed to retrieve relevant legal text, rerank the results, and generate answers from a local LLM while providing source citations.

## What this project does

- Loads `.txt` and `.pdf` documents from the `docs/` directory.
- Splits large documents into smaller context chunks.
- Generates embeddings using the Ollama `bge-m3` model.
- Stores those embeddings in a Chroma vector store at `db/chroma_db`.
- Retrieves relevant text chunks when a user asks a question.
- Reranks retrieved documents using Jina AI's reranker API.
- Sends the reranked context to a local Ollama LLM (`llama3.2`) to generate answers.
- Presents a Streamlit chat UI for question entry and answer streaming.
- Displays source metadata with links to the original PDF page.

## Project structure

- `app.py` - Streamlit application for the chat interface and source display.
- `ingestion_pipeline.py` - Document loading, splitting, and vector store creation.
- `retrieval_pipeline.py` - Retrieval logic, reranking, prompt construction, and answer streaming.
- `run.py` - Launcher script that starts the FastAPI docs server and Streamlit app.
- `server.py` - FastAPI app used to serve the `docs/` directory as static files.
- `docs/` - Default document folder for ingestion.
- `db/chroma_db/` - Persistent Chroma vector store output location.

## Dependencies

The project uses the following Python packages:

- `langchain`
- `langchain-core`
- `langchain-community`
- `langchain-text-splitters`
- `langchain-chroma`
- `langchain-ollama`
- `chromadb`
- `ollama`
- `torchvision`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Setup

1. Place your legal documents in the `docs/` directory.
   - Supported formats: `.txt` and `.pdf`
2. Build the vector store by running:

```bash
python ingestion_pipeline.py
```

If `db/chroma_db` already exists, the ingestion script will reuse the existing vector store.

## Usage

Start the app with:

```bash
python run.py
```

This will:

- Launch the FastAPI static file server for `docs/` on `http://127.0.0.1:8000`
- Launch the Streamlit UI for the chat assistant

Then open the Streamlit UI in your browser.

Alternatively, you can run just the Streamlit app directly:

```bash
python -m streamlit run app.py
```

> Note: `app.py` also tries to start a local HTTP server on port `8000` to serve document files.

## How it works

### Ingestion

- `load_documents()` loads all `.txt` files from `docs/` recursively and `.pdf` files using `PyPDFLoader`.
- `split_documents()` breaks documents into overlapping chunks using `RecursiveCharacterTextSplitter`.
- `create_vector_store()` computes embeddings with `OllamaEmbeddings(model="bge-m3")` and stores chunks in Chroma.

### Retrieval and answer generation

- `retrieve_documents()` performs similarity search against the Chroma store and returns the top 20 candidates.
- `rerank_documents()` calls Jina AI reranker to select the top 5 most relevant documents.
- `build_context()` assembles retrieved text and metadata into a single prompt context.
- `build_messages()` constructs a system + human prompt to instruct the LLM as a legal assistant.
- `ask_question()` streams the local Ollama model response while returning retrieved source documents.

### Streamlit UI

- Users enter questions in a chat box.
- The app streams responses from `ask_question()`.
- It shows source documents with page metadata and provides direct links to the PDF source.

## Important notes

- The system is configured for legal question answering and emphasizes using only retrieved document context.
- The prompt instructs the model to avoid inventing legal facts and to reply explicitly when the answer is unavailable.
- The `JINA_API_KEY` is currently stored in `retrieval_pipeline.py`; for production use, move it to environment variables.
- `run.py` assumes `uvicorn` is available and uses `server.py` to serve `docs/`.

## Extending the project

- Add more legal documents to `docs/` or create a custom ingestion path.
- Replace `OllamaEmbeddings` or the LLM model name if you have another Ollama model installed.
- Improve prompt engineering in `retrieval_pipeline.py` for better legal answer quality.

