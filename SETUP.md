# Setup Guide

This repository contains two scripts:
- `test_qwen.py` (Ollama + Qwen model)
- `test_transformer.py` (Weaviate + text2vec-transformers)

Below are step-by-step setup commands for Windows PowerShell.

## 1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install ollama weaviate-client sentence-transformers numpy pandas
```

## 3) Ollama setup (for `test_qwen.py`)

Make sure Ollama is installed and running, then pull the model used by the script:

```powershell
ollama pull qwen2.5:7b
```

Run the script:

```powershell
python .\test_qwen.py
```

## 4) Qdrant setup

Start Qdrant:

```powershell
docker compose up -d
```

Wait until Qdrant is healthy (optional but helpful):

```powershell
curl http://localhost:6333/healthz
```

## 5) Stop services

```powershell
docker compose down
```
