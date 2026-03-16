# Setup Guide

This repository contains three scripts (under `tests/`):
- `tests/test_reasoning_model.py` (Ollama + Qwen model)
- `tests/test_vector_db.py` (Qdrant + sentence-transformers)
- `tests/test_rag_chat.py` (Ollama + Qdrant + hybrid rerank)

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

## 3) Ollama setup (for `tests/test_reasoning_model.py`)

Make sure Ollama is installed and running, then pull the model used by the script:

```powershell
ollama pull qwen2.5:7b
```

Run the script:

```powershell
python .\tests\test_reasoning_model.py
```

## 4) Qdrant setup (for `tests/test_vector_db.py` and `tests/test_rag_chat.py`)

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
python .\tests\test_rag_chat.py
```

## 5) Stop services

```powershell
docker compose down
```
