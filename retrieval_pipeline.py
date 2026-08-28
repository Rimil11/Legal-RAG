import os
import time
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
import requests

# ======================================================
# Configuration
# ======================================================

PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "llama3.2"
# LLM_MODEL = "qwen3:4b"




# ======================================================
# Load Models
# ======================================================

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

llm = OllamaLLM(model=LLM_MODEL)

# Load the reranker ONCE

JINA_API_KEY = "jina_49b1c9ee41344667a4e180eccba1ddf389-d4bBOcePW8uIdxAhHs7hmYNXQ"

def rerank_documents(question, docs, top_n=5):

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": question,
            "documents": [doc.page_content for doc in docs],
            "top_n": top_n,
            "return_documents": False
        }
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
    print(f"\nFound {len(docs)} relevant documents.\n")

    sources = []

    for index, doc in enumerate(docs, start=1):

        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")

        sources.append((source, page))

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

        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")

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
Try to think as response as a Legal Assistant (Never misguide with wrong knowledge).

Rules:
1. Never use outside knowledge.
2. Never invent legal facts.
3. If the answer is unavailable, reply exactly:
"I couldn't find the answer in the provided legal documents."
4. Combine multiple retrieved documents when helpful.
5. Always mention document name, section/article number and page number whenever available.
6. Keep answers clear and concise.
6. When answering:
    - Always use ONLY the metadata provided with each retrieved document.
    - Never infer or guess page numbers, section numbers, article numbers, or document names.
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

        document = source.get("document", "Unknown")
        page = source.get("page", "N/A")
        section = source.get("section")
        title = source.get("title")
        key = (document, page, section)
        
        if key in shown:
            continue

        shown.add(key)
        print(f"- Document : {document}")

        if section:
            print(f"  Section  : {section}")

        if title:
            print(f"  Title    : {title}")
        print(f"  Page     : {page}")
        print("--------------------------------------")


def typewriter(text, delay=0.01):

    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)

    print()


# ===============================
# Main RAG Function
# ======================================================

def ask_question(question):

    docs = retrieve_documents(question)
    context = build_context(docs)
    messages = build_messages(question, context)

    def stream():

        for chunk in llm.stream(messages):
            yield chunk

    return stream(), docs

# ======================================================
# Chat Loop
# ======================================================

def start_chat():

    print("Legal Assistant")
    print("Type 'quit' to exit.\n")

    while True:

        question = input("Your Question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        
        response = ask_question(question)
        typewriter(response)


if __name__ == "__main__":
    start_chat()

