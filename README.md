# Talk-to-PDF: Interactive PDF Question-Answering System

## Overview

**Talk-to-PDF** is an interactive system that allows users to upload a PDF and ask questions or continue a discussion based on its content. The system leverages modern NLP techniques including tokenization, embeddings, transformer-based encoders, and Masked Language Modeling (MLM) to generate context-aware answers and demonstrate next-word prediction capabilities.

---

## Features

1. **PDF Upload & Processing**

   - Accepts PDF documents from the user.
   - Splits content into manageable chunks for semantic understanding.
   - Stores embeddings in a vector database (Pinecone) for retrieval.

2. **Interactive Q&A**

   - Users can ask questions about the PDF content.
   - System retrieves relevant chunks using embeddings and a RAG-style pipeline.
   - Generates answers using a local LLM (Ollama `qwen2.5`).

3. **NLP Concepts Implemented**

   - **Tokenizer:** Converts text into sequences of tokens (integers).
   - **Embedding:** Maps discrete tokens into dense vector representations capturing semantic meaning.
   - **Encoder (Transformer):** Processes embeddings via self-attention for contextual understanding.
   - **Task Head:** Uses encoder outputs for token generation (text completion) or classification (MLM).

4. **Masked Language Modeling (MLM) Demo**
   - Demonstrates next-word prediction using a BERT-style architecture.
   - Allows masked tokens in sentences extracted from PDFs.
   - Predicts masked tokens based on context to show model understanding.

---

## Architecture

```text
PDF Document
    │
    ▼
[PDF Split & Chunking]
    │
    ▼
[Embedder] --> Pinecone Index
    │
    ▼
[User Query]
    │
    ▼
[Retriever: Top-K Chunks]
    │
    ▼
[Prompt Builder]
    │
    ▼
[LLM: Ollama / Gemini]
    │
    ▼
[Answer]
```

MLM Pipeline: PDF → Chunk → Mask Token → BERT → Predict → Display

---

## Installation

```bash
# Clone repository
git clone <repo_url>
cd Talk-to-PDF
```

### Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add

#Ingest the PDF first

```bash
python .\ingestion\pipeline.py
```

### Start FastAPI backend

```bash
uvicorn main:app --reload
```

### Start Streamlit frontend

```bash
streamlit run app.py
```

---

## Usage

- Run Ingest istruction to ingest the input pdf andstore embeddings in Pinecone.

- Open StreamLit for the chat function or see the MLM demo.

- Chat Function

  - Enter your question in the text input.

  - Click Get Answer to retrieve the answer using the RAG pipeline.

  -

- MLM Demo

  - Select a sentence from the PDF.

  - Mask a token to predict using BERT.

  - Click Run MLM demo to see next-word prediction.

---
