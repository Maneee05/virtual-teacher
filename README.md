# 🔍 RAG Pipeline — Virtual Teacher (Collaborative Project)

> **Note:** This repository contains the **RAG pipeline component** I built as part of a collaborative academic mini-project called Virtual Teacher. The broader application (UI, avatar integration, full system) is being developed separately by teammates. This repo covers my contribution: the retrieval and generation pipeline.

---

## 🛠️ What I Built

A **Retrieval-Augmented Generation (RAG) pipeline** that:

1. Ingests curriculum documents (PDFs)
2. Chunks and embeds the content using SentenceTransformers
3. Stores vectors in a **Qdrant** vector database
4. Builds a **NetworkX knowledge graph** to map topic relationships
5. Performs **hybrid retrieval** — combining semantic vector search with graph-based topic linking
6. Passes retrieved context to **Gemini LLM** to generate structured lesson plans

The hybrid retrieval approach (vector + graph) is the key design decision — it reduces the context-gap problem common in naive RAG, where a vector match alone can miss conceptually related topics.

---

## 🧱 Pipeline Architecture

```text
Curriculum PDF
      ↓
PDF Parsing  (PyMuPDF)
      ↓
Text Chunking  (LangChain)
      ↓
Embedding Generation  (SentenceTransformers — all-MiniLM-L6-v2)
      ↓
   ┌──────────────────────────────────────┐
   │  Qdrant Vector Store                 │
   │  NetworkX Knowledge Graph            │
   └──────────────────────────────────────┘
            ↓  Hybrid Retrieval
   Relevant context chunks + related topics
            ↓
   Gemini LLM  (via OpenAI-compatible API)
            ↓
   Structured Lesson Plan Output
   (Learning objectives · Explanation · Visual aids)
```

---

## ⚙️ Tech Stack

| Component         | Technology                              |
|-------------------|-----------------------------------------|
| Language          | Python                                  |
| Text Processing   | LangChain                               |
| Embeddings        | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Database   | Qdrant                                  |
| Knowledge Graph   | NetworkX                                |
| PDF Parsing       | PyMuPDF                                 |
| LLM               | Gemini (via OpenAI-compatible API)      |

---

## 📁 Repository Structure

```text
virtual-teacher/
│
├── data/
│   └── .env                # API keys (not committed)
├── rag-pipeline/
│   ├── src.py              # Core RAG pipeline
│   └── testing.py          # Pipeline tests and example queries
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

### 1. Clone and install

```bash
git clone https://github.com/Maneee05/virtual-teacher.git
cd virtual-teacher
pip install -r requirements.txt
```

### 2. Add your API key

Create a `.env` file inside `/data`:

```env
GEMINI_API_KEY=your_api_key_here
```

### 3. Run the pipeline

```bash
cd rag-pipeline
python src.py
```

---

## 📌 Example Output

**Query:** `"What is TCP/IP?"`

```text
Learning Objectives:
- Understand the TCP/IP communication model
- Learn how packets are routed across networks

Explanation:
TCP/IP is a layered communication protocol...

Visual Aids:
- Layered network model diagram
- Packet flow animation
```

---

## 🧠 Design Decisions

**Why hybrid retrieval?**
Pure vector search retrieves semantically similar chunks but can miss conceptually related topics that aren't lexically similar. Layering a knowledge graph lets the retriever follow topic relationships (e.g. "TCP/IP" → "OSI model" → "network layers") even when the query doesn't mention them explicitly. This improves answer completeness for educational content.

**Why Qdrant over FAISS/ChromaDB?**
Qdrant supports filtered search and scales well without needing to reload the index — more practical for a modular pipeline design where documents are added incrementally.

---

## 👩‍💻 Author

**Maneesha Manohar** — B.Tech CSE @ CUSAT  
[LinkedIn](https://linkedin.com/in/maneesha-manohar-607819249) · [GitHub](https://github.com/Maneee05)
