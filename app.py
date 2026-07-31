import os
import socket
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import streamlit as st
from retrieval_pipeline import ask_question


# ---------------- PDF HTTP SERVER ----------------

def start_pdf_server(port=8000):
    """Starts a background HTTP server only once."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect(("127.0.0.1", port))
        sock.close()
        return
    except OSError:
        pass

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                directory=os.getcwd(),      # serves your project folder
                **kwargs
            )

    def run():
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server.serve_forever()

    threading.Thread(target=run, daemon=True).start()


# ---------------- STREAMLIT ----------------

st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

start_pdf_server()

st.title("⚖️ Legal AI Assistant")
st.write("Ask questions about Indian laws.")


# ---------------- SESSION ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "docs" not in st.session_state:
    st.session_state.docs = []


# ---------------- CHAT HISTORY ----------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------- CHAT INPUT ----------------

question = st.chat_input("Ask a legal question...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching legal documents..."):

            stream, docs = ask_question(question)

            answer = st.write_stream(stream)

        st.session_state.docs = docs

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        })


# ---------------- SOURCES ----------------

shown = set()

for doc in st.session_state.docs:

    pdf = doc.metadata.get("source")
    page = doc.metadata.get("page")

    key = (pdf, page)

    if not pdf or key in shown:
        continue

    shown.add(key)

    with st.expander(f"📄 {pdf} (Page {page})"):

        st.write(f"**Document:** {pdf}")
        st.write(f"**Page:** {page}")

        pdf_url = f"http://127.0.0.1:8000/docs/{pdf}#page={page}"

        st.link_button(
            "📄 Open Source",
            pdf_url,
            use_container_width=True,
            key=f"{pdf}-{page}"
        )