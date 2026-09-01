import os
import time
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests
import streamlit as st

load_dotenv()


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


JINA_API_KEY = get_secret("JINA_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")
# ======================================================
# Configuration
# ======================================================

load_dotenv()

PERSIST_DIRECTORY = "db/chroma_db"

EMBEDDING_MODEL = "jina-embeddings-v5-text-small"
LLM_MODEL = "openai/gpt-oss-20b"

JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"

# JINA_API_KEY = os.getenv("JINA_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ======================================================
# Jina Embeddings
# ======================================================

class JinaEmbeddings:

    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model

    def embed_query(self, text):

        response = requests.post(
            JINA_EMBEDDING_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "input": [text],
                "task": "retrieval.query",
                "dimensions": 1024,
                "normalized": True,
                "embedding_type": "float"
            },
            timeout=60
        )

        response.raise_for_status()

        return response.json()["data"][0]["embedding"]

    def embed_documents(self, texts):

        response = requests.post(
            JINA_EMBEDDING_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "input": texts,
                "task": "retrieval.passage",
                "dimensions": 1024,
                "normalized": True,
                "embedding_type": "float"
            },
            timeout=180
        )

        response.raise_for_status()

        data = response.json()["data"]

        data.sort(
            key=lambda item: item["index"]
        )

        return [
            item["embedding"]
            for item in data
        ]


# ======================================================
# Validate API Keys
# ======================================================

if not JINA_API_KEY:
    raise ValueError(
        "JINA_API_KEY is not set. "
        "Add it to your .env file."
    )

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file."
    )


# ======================================================
# Load Embeddings
# ======================================================

embeddings = JinaEmbeddings(
    api_key=JINA_API_KEY,
    model=EMBEDDING_MODEL
)


# ======================================================
# Load ChromaDB
# ======================================================

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)


# ======================================================
# Load LLM
# ======================================================

llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
    max_tokens=2048
)


# ======================================================
# Rerank Documents
# ======================================================

def rerank_documents(question, docs, top_n=5):

    response = requests.post(
        JINA_RERANK_URL,
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": question,
            "documents": [
                doc.page_content
                for doc in docs
            ],
            "top_n": top_n,
            "return_documents": False
        },
        timeout=60
    )

    response.raise_for_status()

    results = response.json()["results"]

    reranked_docs = [
        docs[item["index"]]
        for item in results
    ]

    return reranked_docs


# ======================================================
# Helper Functions
# ======================================================

def retrieve_documents(question):

    docs = db.similarity_search(
        question,
        k=20
    )

    docs = rerank_documents(
        question,
        docs,
        top_n=5
    )

    return docs


def print_documents(docs):

    print(
        f"\nFound {len(docs)} relevant documents.\n"
    )

    sources = []

    for index, doc in enumerate(
        docs,
        start=1
    ):

        source = os.path.basename(
            doc.metadata.get(
                "source",
                "Unknown"
            )
        )

        page = doc.metadata.get(
            "page",
            "N/A"
        )

        sources.append(
            (source, page)
        )

        print("=" * 70)
        print(f"Document {index}")
        print(f"Source : {source}")
        print(f"Page   : {page}")
        print("-" * 70)

        # print(doc.page_content)

    return sources


def build_context(docs):

    context = []

    for doc in docs:

        source = os.path.basename(
            doc.metadata.get(
                "source",
                "Unknown"
            )
        )

        page = doc.metadata.get(
            "page",
            "N/A"
        )

        context.append(
            f"""
Document: {source}
Page: {page}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(context)


def build_messages(question, context):

    return [

        SystemMessage(
            content="""
You are an AI Legal Assistant.

Answer from the context of retrieved documents.
Try to think as response as a Legal Assistant
(Never misguide with wrong knowledge).

Rules:
1. Never use outside knowledge.
2. Never invent legal facts.
3. If the answer is unavailable, reply exactly:
"I couldn't find the answer in the provided legal documents."
4. Combine multiple retrieved documents when helpful.
5. Always mention document name, section/article number
   and page number whenever available.
6. Keep answers clear and concise.
7. When answering:
    - Always use ONLY the metadata provided with each
      retrieved document.
    - Never infer or guess page numbers, section numbers,
      article numbers, or document names.
    - If metadata is missing, omit it rather than inventing it.
"""
        ),

        HumanMessage(
            content=f"""
Retrieved Context
=================

{context}

=================

Question:
{question}
"""
        )
    ]


def print_sources(sources):

    print("\nSources:")

    shown = set()

    for source in sources:

        document = source.get(
            "document",
            "Unknown"
        )

        page = source.get(
            "page",
            "N/A"
        )

        section = source.get(
            "section"
        )

        title = source.get(
            "title"
        )

        key = (
            document,
            page,
            section
        )

        if key in shown:
            continue

        shown.add(key)

        print(
            f"- Document : {document}"
        )

        if section:
            print(
                f"  Section  : {section}"
            )

        if title:
            print(
                f"  Title    : {title}"
            )

        print(
            f"  Page     : {page}"
        )

        print(
            "--------------------------------------"
        )


def typewriter(text, delay=0.01):

    for char in text:

        print(
            char,
            end="",
            flush=True
        )

        time.sleep(delay)

    print()


# ======================================================
# Main RAG Function
# ======================================================

def ask_question(question):

    docs = retrieve_documents(
        question
    )

    context = build_context(
        docs
    )

    messages = build_messages(
        question,
        context
    )

    def stream():

        for chunk in llm.stream(
            messages
        ):

            if chunk.content:
                yield chunk.content

    return stream(), docs


# ======================================================
# Chat Loop
# ======================================================

def start_chat():

    print(
        "Legal Assistant"
    )

    print(
        "Type 'quit' to exit.\n"
    )

    while True:

        question = input(
            "Your Question: "
        )

        if question.lower() == "quit":

            print(
                "Goodbye!"
            )

            break

        try:

            response, docs = ask_question(
                question
            )

            print()

            for chunk in response:

                print(
                    chunk,
                    end="",
                    flush=True
                )

            print("\n")

            print("Sources:")

            for doc in docs:

                source = os.path.basename(
                    doc.metadata.get(
                        "source",
                        "Unknown"
                    )
                )

                page = doc.metadata.get(
                    "page",
                    "N/A"
                )

                print(
                    f"- {source}, Page {page}"
                )

            print()

        except Exception as e:

            print(
                f"\nError: {e}\n"
            )


if __name__ == "__main__":
    start_chat()
