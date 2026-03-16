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

## 4) Weaviate setup (for `test_transformer.py`)

`test_transformer.py` expects a local Weaviate instance with the
`text2vec-transformers` module enabled.

### Option A: Docker (example)

Create a `docker-compose.yml` in this folder with the following content:

```yaml
version: "3.4"
services:
  weaviate:
    image: semitechnologies/weaviate:1.24.4
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: "/var/lib/weaviate"
      DEFAULT_VECTORIZER_MODULE: "text2vec-transformers"
      ENABLE_MODULES: "text2vec-transformers"
      CLUSTER_HOSTNAME: "node1"
  t2v-transformers:
    image: semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
    environment:
      ENABLE_CUDA: "0"
```

Start Weaviate:

```powershell
docker compose up -d
```

Run the script:

```powershell
python .\test_transformer.py
```

### Option B: Existing Weaviate instance

If you already run Weaviate, ensure it is listening at `http://localhost:8080`
and has `text2vec-transformers` enabled, then run:

```powershell
python .\test_transformer.py
```

## 5) Stop services

```powershell
docker compose down
```
