# ⚖️ Legal RAG AI Assistant

> An AI-powered Legal Assistant that answers questions from legal documents using **Retrieval-Augmented Generation (RAG)**, **LangChain**, **ChromaDB**, **Ollama**, and **Streamlit**.

---

## 📖 Overview

Legal documents are often lengthy and difficult to navigate. This project enables users to ask legal questions in natural language and receive context-aware answers backed by relevant legal document citations.

The system retrieves the most relevant document chunks, reranks them for better accuracy, and generates responses using a local Large Language Model (LLM), reducing hallucinations by grounding answers in retrieved context.

---

## ✨ Features

* 📄 Supports PDF and TXT legal documents
* 🔍 Semantic search using Chroma Vector Database
* 🧠 Local embeddings with **bge-m3**
* 🤖 Local LLM inference with **llama3.2**
* 📑 Intelligent document chunking
* ⚡ Reranking using Jina AI
* 💬 Interactive Streamlit chat interface
* 📚 Source citations with page numbers
* 🔗 Direct links to original documents

---

## 🏗️ Architecture

```text
Legal Documents
       │
       ▼
Document Loader
       │
       ▼
Text Chunking
       │
       ▼
Embeddings (bge-m3)
       │
       ▼
ChromaDB Vector Store
       │
 User Question
       │
       ▼
Similarity Search
       │
       ▼
Jina AI Reranker
       │
       ▼
LLM (llama3.2)
       │
       ▼
Answer + Source Citations
```

---

## 🚀 Tech Stack

| Category        | Technologies      |
| --------------- | ----------------- |
| Language        | Python            |
| Framework       | LangChain         |
| LLM             | Ollama (llama3.2) |
| Embeddings      | Ollama (bge-m3)   |
| Vector Database | ChromaDB          |
| UI              | Streamlit         |
| Backend         | FastAPI           |
| Reranker        | Jina AI           |

---

## 📂 Project Structure

```text
Legal-RAG/
│── app.py
│── ingestion_pipeline.py
│── retrieval_pipeline.py
│── run.py
│── server.py
│── requirements.txt
│── README.md
│
├── docs/
└── db/
    └── chroma_db/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Legal-RAG.git
cd Legal-RAG
```

Create a virtual environment:

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS**

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

Replace the **📥 Prepare Documents** section with the following:

## 📥 Prepare Documents

Place your legal documents inside the `docs/` directory.

Supported formats:

* PDF
* TXT

Build the vector database:

```bash
python ingestion_pipeline.py
```

> **📝 Note**
>
> * To ingest a **new or different set of documents**, first delete the existing **`db/chroma_db/`** folder.
> * If the `chroma_db` folder already exists, the application will reuse the existing vector database instead of creating a new one.
> * After deleting the folder, rebuild the vector database by running:
>
> ```bash
> python ingestion_pipeline.py
> ```


```bash
python ingestion_pipeline.py
```

---

## ▶️ Run the Application

Start the application using:

```bash
python run.py
```

The `run.py` script automatically launches:

* **FastAPI Server (`server.py`)** – Serves the `docs/` directory, enabling PDF source previews directly from the application.
* **Streamlit Application (`app.py`)** – Starts the Legal RAG chat interface.

Using `run.py` is the recommended approach, as both services are required for the complete application experience, including **interactive PDF source previews**.

If you only want to run the chat interface without PDF preview support:

```bash
streamlit run app.py
```


---

## 💡 How It Works

1. Load legal documents.
2. Split documents into semantic chunks.
3. Generate embeddings and store them in ChromaDB.
4. Retrieve relevant chunks for user queries.
5. Rerank results using Jina AI.
6. Generate grounded responses with a local LLM.
7. Display answers with source citations and page numbers.

---

## 📸 Screenshots

Add screenshots here.

* Home Page
* Chat Interface
* Source Citations
* PDF Viewer

---

## 🚀 Future Improvements

* Replace local models with larger cloud-hosted LLMs (GPT, Claude, Gemini, etc.) for improved reasoning.
* Build a modern **React.js** frontend with a **FastAPI** backend instead of Streamlit.
* Implement Hybrid Search (Dense + BM25) for better retrieval accuracy.
* Add conversation memory and multilingual support.
* Highlight cited text directly inside PDFs.
* Support OCR for scanned legal documents.
* Containerize the application with Docker and deploy it to the cloud.
* Add user authentication, analytics, and performance monitoring.
* Support **scenario-based legal queries** by identifying applicable legal provisions from real-world situations (e.g., *"Suppose a man beats his wife. What charges may apply?"*).

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push your branch.
5. Open a Pull Request.

---

## 👨‍💻 Author

**Rimil Hans**

If you found this project useful, consider giving it a ⭐ on GitHub.
