# Agentic Self-Hosted Modular RAG

Local, self-hosted Retrieval-Augmented Generation (RAG) pipeline with modular components for ingestion, hybrid retrieval (dense + BM25), reranking, and LangGraph orchestration. Designed to run on a laptop while keeping a production-style structure.

## Architecture
```mermaid
flowchart LR
  A[Markdown Files] --> B[Chunking + Ingestion]
  B --> C[Vector Index (Qdrant)]
  B --> D[BM25 Index]
  C --> E[Hybrid Retrieval]
  D --> E
  E --> F[Reranker]
  F --> G[Prompt + LLM Answer]
```

## Highlights
- Hybrid retrieval: Qdrant vectors + BM25 score fusion
- Reranking with a cross-encoder
- Query rewrite + simple intent routing (direct vs RAG vs summarize)
- LangGraph pipeline with local or LangSmith traces
- Fully local LLM inference via Ollama

## Changelog

### Next Steps
- Add real unit tests with pytest (router, chunker, retrieval scoring).
- Add a lightweight CI workflow to run unit tests on push.
- Add optional integration tests gated on Qdrant/Ollama availability.
- Add ingestions of other file types (pdf, xlsx, ppx, word)
- Add UI
- Add chat memory

### 2026-03-18
- RAG agentic worflow (router + retrieval + answer nodes) as a structured LangGraph flow.
- Markdown data (GoodWiki) for testing
- Logging with LangSmith for debugging
- Hybrid search swith score fusion across dense (Qdrant) and BM25 retrieval.
- Reranking (BAAI/bge-reranker-base) with a cross-encoder for improved top-k quality.

### 2026-03-16
- Setup LLM (qwen2.5:7b) and embedding (BAAI/bge-small-en-v1.5) models 
- Set-up vector DB (Qdrant) Docker configuration for local run
- Added LLM inference via Ollama for local run


## Project Structure
- `app/` runtime code (ingestion, retrieval, RAG pipeline, agents)
- `scripts/` data download and sampling helpers
- `data/` local datasets and indexes (git-ignored)
- `tests/` interactive demos and eval scripts

## Requirements
- Python 3.10+
- Docker (for Qdrant)
- Ollama installed and running

## Quickstart
1. Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
3. Start Qdrant:
```powershell
docker compose up -d
```
4. Pull the Ollama model:
```powershell
ollama pull qwen2.5:7b
```
5. Prepare data (see next section), then run:
```powershell
python -m app.rag.pipeline
```

## Data: Download or Provide Markdown
By default, `DATA_DIR` points to `data/goodwiki_markdown_sample` (see `app/utils/config.py`). The `data/` folder is git-ignored, so you need local content.

Option A: Download GoodWiki markdown and create a sample:
```powershell
python scripts/goodwiki_data.py
```
This creates `data/goodwiki_markdown/` and samples into `data/goodwiki_markdown_sample/`.

Option B: Use your own markdown:
- Place `.md` files in a local folder
- Update `DATA_DIR` in `app/utils/config.py`

## Run the LangGraph Pipeline
```powershell
python -m app.rag.pipeline
```
Routing behavior:
- `direct`: simple chat
- `rag`: retrieval-augmented answer
- `summarize`: retrieval + summary

Tracing:
- If `LANGGRAPH_USE_LANGSMITH_API = False`, traces are written to `data/local_traces.jsonl`
- To use LangSmith, set `LANGGRAPH_USE_LANGSMITH_API = True` and configure `.env` (see `.env.example`)

## Configuration
Core settings live in `app/utils/config.py`:
- Models: `EMBED_MODEL_NAME`, `RERANKER_MODEL_NAME`, `LLM_MODEL_NAME`
- Retrieval: `DENSE_ALPHA`, `RETRIEVE_K`, `RERANK_K`
- Context assembly: `CONTEXT_MAX_TOKENS`, `CONTEXT_MAX_DOCS`, `DEDUPLICATE_BY_FILE`
- Query rewrite / intent: `ENABLE_QUERY_REWRITE`, `ENABLE_INTENT_DETECTION`
- Data paths and tracing: `DATA_DIR`, `STATE_PATH`, `LOCAL_TRACE_PATH`

## Interactive Demos / Tests
- `tests/test_langsmith_eval.py` evaluation harness (writes CSVs to `data/evals/`)

## Stop Services
```powershell
docker compose down
```

## Troubleshooting
- `No markdown documents found`: verify `DATA_DIR` and that it contains `.md` files
- `Qdrant connection error`: ensure `docker compose up -d` is running and port `6333` is reachable
- `Ollama not responding`: make sure the Ollama service is running and the model is pulled



