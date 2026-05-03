# 🎓 Virtual Teacher – Multimodal RAG Learning System

An intelligent **Retrieval-Augmented Generation (RAG)** based virtual teacher that generates structured lesson plans from curriculum materials using semantic search, knowledge graphs, and LLMs.

---

## 🚀 Overview

Virtual Teacher is a **mini-project focused on AI-powered education**, designed to simulate a personalized teaching assistant.

The system ingests learning materials (PDFs, images), builds a **semantic + graph-based knowledge base**, and generates **context-aware lesson plans** for students using LLMs.

---

## ✨ Key Features

* 📚 **Multimodal Input Support**

  * PDF parsing (PyMuPDF)
  * Image handling (extensible for OCR)

* 🧠 **RAG Pipeline**

  * Text chunking (LangChain)
  * Embeddings (Sentence Transformers)
  * Vector storage (Qdrant)

* 🔎 **Hybrid Retrieval**

  * Semantic search (vector similarity)
  * Knowledge graph-based topic linking (NetworkX)

* 🧩 **Knowledge Graph Construction**

  * Automatic topic extraction
  * Relationship mapping between concepts

* 🤖 **LLM Integration**

  * Gemini (via OpenAI-compatible API)
  * Generates structured lesson plans:

    * Learning objectives
    * Explanation
    * Visual aids

---

## 🏗️ System Architecture

```
User Query
     ↓
Hybrid Retriever (Qdrant + Knowledge Graph)
     ↓
Relevant Context Retrieval
     ↓
LLM (Gemini API)
     ↓
Structured Lesson Plan Output
```

---

## 🛠️ Tech Stack

| Component       | Technology                              |
| --------------- | --------------------------------------- |
| Embeddings      | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector DB       | Qdrant                                  |
| Text Processing | LangChain                               |
| Knowledge Graph | NetworkX                                |
| PDF Processing  | PyMuPDF                                 |
| LLM             | Gemini (OpenAI-compatible API)          |
| Language        | Python                                  |

---

## 📂 Project Structure

```
virtual-teacher/
│
├── data/
│    └── .env               # API keys
├── rag-pipeline/
│   └── src.py             # Main RAG pipeline
│   └── testing.py         # Implementation scripts
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Maneee05/virtual-teacher.git
cd virtual-teacher
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

---

## ▶️ Usage

### Run the pipeline

```bash
cd code-rag
python src.py
```

### Example Workflow

1. Upload curriculum files (PDF/images)
2. System processes and builds knowledge base
3. Enter student query:

```
"What is TCP/IP?"
```

4. Output:

* Learning objectives
* Explanation
* Suggested visual aids

---

## 📊 Example Output

```
Learning Objectives:
- Understand TCP/IP layers
- Learn packet communication basics

Explanation:
TCP/IP is a communication model...

Visual Aids:
- 3D layered network model
- Packet flow animation
```

---

## 🧠 Design Highlights (For Recruiters)

* Combines **vector search + graph-based reasoning**
* Implements **hybrid retrieval for improved relevance**
* Uses **modular pipeline design** (ingestion → indexing → retrieval → generation)
* Demonstrates **applied NLP + IR concepts**
* Easily extensible to:

  * Voice-based teaching
  * Real-time tutoring
  * 3D avatar integration

---

## 🚧 Future Improvements

* Real OCR for images (Tesseract / Vision models)
* Frontend UI (React / Flutter)
* 3D avatar integration for teaching
* Personalized learning paths
* Fine-tuned LLM for education domain

---

## 👩‍💻 Author

Maneesha Manohar

B.Tech CSE Student | AI/ML Enthusiast

Linkedin - www.linkedin.com/in/maneesha-manohar-607819249

---

## ⭐ Why This Project Stands Out

This project goes beyond basic chatbots by integrating:

* Retrieval-Augmented Generation
* Knowledge Graphs
* Multimodal learning inputs

It reflects strong understanding of **modern AI system design**, not just model usage.

---
