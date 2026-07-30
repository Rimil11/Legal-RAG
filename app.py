import base64
import streamlit as st
from retrieval_pipeline import ask_question

st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal AI Assistant")
st.write("Ask questions about Indian laws.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="800px"
        type="application/pdf">
    </iframe>
    """

    st.markdown(pdf_display, unsafe_allow_html=True)


# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input("Ask a legal question...")

if question:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching legal documents..."):

            stream, docs = ask_question(question)
            answer = st.write_stream(stream)

        st.divider()
        st.subheader("📚 Source Documents")

        shown = set()

        for doc in docs:

            pdf_name = doc.metadata.get("source")

            if not pdf_name or pdf_name in shown:
                continue

            shown.add(pdf_name)

            page = doc.metadata.get("page", "Unknown")

            with st.expander(f"📄 {pdf_name} (Page {page})"):

                st.write(f"**Document:** {pdf_name}")
                st.write(f"**Page:** {page}")

                if st.button(
                    f"Open {pdf_name}",
                    key=f"{pdf_name}_{page}"
                ):
                    show_pdf(f"docs/{pdf_name}")

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )