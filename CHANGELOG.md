# Changelog

## Next Steps
- Add real unit tests with pytest (router, chunker, retrieval scoring).
- Add a lightweight CI workflow to run unit tests on push.
- Add optional integration tests gated on Qdrant/Ollama availability.
- Add ingestions of other file types (pdf, xlsx, ppx, word)
- Add UI
- Add chat memory

## 2026-03-18
- Made the pipeline LangGraph-only with a single entry point in `app/rag/pipeline.py`.
- Consolidated GoodWiki download and sampling into `scripts/goodwiki_data.py`.
- Updated README with architecture diagram and setup details.
