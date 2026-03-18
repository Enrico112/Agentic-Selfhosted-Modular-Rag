# Setup Guide

This repository contains four scripts (under `tests/`):
- `tests/test_llm.py` (Ollama + Qwen model)
- `tests/test_vector_db.py` (Qdrant + sentence-transformers)
- `tests/test_rag_pipeline.py` (Ollama + Qdrant + hybrid rerank)
- `tests/test_chunking.py` (Markdown chunking over sample data)

Below are step-by-step setup commands for Windows PowerShell.

## 1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3) Ollama setup (for `tests/test_llm.py`)

Make sure Ollama is installed and running, then pull the model used by the script:

```powershell
ollama pull qwen2.5:7b
```

Run the script:

```powershell
python .\tests\test_llm.py
```

## 4) Qdrant setup (for `tests/test_vector_db.py` and `tests/test_rag_pipeline.py`)

Start Qdrant:

```powershell
docker compose up -d
```

Wait until Qdrant is healthy (optional but helpful):

```powershell
curl -UseBasicParsing http://localhost:6333/healthz
```

Run the script:

```powershell
python .\tests\test_vector_db.py
```

Run the RAG chat script:

```powershell
python .\tests\test_rag_pipeline.py
```

## 5) Markdown chunking test (optional)

This uses `data/goodwiki_markdown_sample` as input (see `tests/sample_goodwiki.py`).

```powershell
python .\tests\test_chunking.py
```

## 6) Run the main RAG pipeline

```powershell
python -m app.rag.pipeline
```

## 7) Stop services

```powershell
docker compose down
```
