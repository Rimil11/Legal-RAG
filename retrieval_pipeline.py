import os
import time
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM


# ======================================================
# Configuration
# ======================================================

PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "llama3.2"
TOP_K = 8




# ======================================================
# Load Models
# ======================================================

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

llm = OllamaLLM(model=LLM_MODEL)


# ======================================================
# Helper Functions
# ======================================================

def retrieve_documents(question):
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    return retriever.invoke(question)


def print_documents(docs):
    print(f"\nFound {len(docs)} relevant documents.\n")

    sources = []

    for index, doc in enumerate(docs, start=1):

        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")

        sources.append((source, page))

        # print("=" * 70)
        # print(f"Document {index}")
        # print(f"Source : {source}")
        # print(f"Page   : {page}")
        # print("-" * 70)
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

Answer ONLY from the retrieved documents.

Rules:
1. Never use outside knowledge.
2. Never invent legal facts.
3. If the answer is unavailable, reply exactly:
"I couldn't find the answer in the provided legal documents."
4. Combine multiple retrieved documents when helpful.
5. Always mention document name, section/article number and page number whenever available.
6. Keep answers clear and concise.
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

    for source, page in sources:

        if (source, page) not in shown:

            page_text = page + 1 if isinstance(page, int) else page
            print(f"- {source} (Page: {page_text})")

            shown.add((source, page))


def typewriter(text, delay=0.01):

    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)

    print()


# ===============================
# Main RAG Function
# ======================================================

def ask_question(question):

    # print("\n" + "=" * 80)
    # print("Searching Documents...")
    # print("=" * 80)

    docs = retrieve_documents(question)

    sources = print_documents(docs)

    context = build_context(docs)

    messages = build_messages(question, context)

    # print("\n" + "=" * 80)
    # print("Generating Answer...")
    # print("=" * 80)

    # answer = llm.invoke(messages)
    # return answer

    for chunk in llm.stream(messages):
        yield chunk
        # print(chunk)

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

