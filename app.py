import os

import streamlit as st
from retrieval_pipeline import ask_question


# ---------------- STREAMLIT ----------------

st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

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
        }
    )


# ---------------- SOURCES ----------------

shown = set()

for doc in st.session_state.docs:

    pdf = doc.metadata.get("source")
    page = doc.metadata.get("page")

    if not pdf:
        continue

    filename = os.path.basename(pdf)

    key = (filename, page)

    if key in shown:
        continue

    shown.add(key)

    # Find PDF inside docs folder
    pdf_path = os.path.join(
        "docs",
        filename
    )

    with st.expander(
        f"📄 {filename} (Page {page})"
    ):

        st.write(
            f"**Document:** {filename}"
        )

        st.write(
            f"**Page:** {page}"
        )

        if os.path.exists(pdf_path):

            # Deployment-compatible GitHub PDF URL
            pdf_url = (
                f"https://raw.githubusercontent.com/"
                f"Rimil11/Legal-RAG/main/{filename}"
                f"#page={page}"
            )

            st.link_button(
                "📄 Open Source PDF",
                pdf_url,
                use_container_width=True
            )

        else:

            st.error(
                f"PDF file not found: {pdf_path}"
            )