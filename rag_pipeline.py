from typing import Dict, List, Tuple, Any
import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from llm import chat, generate_answer
from config import DEBUG
from markdown_chunker import chunk_markdown
from prompts import build_prompt
from retrieval import (
    Document,
    build_bm25_index,
    filter_context,
    hybrid_retrieve,
    index_documents,
    rerank,
)

DATA_DIR = Path("data/goodwiki_markdown_sample")
STATE_PATH = Path("data/.rag_index_state.json")


def rewrite_query(query: str) -> str:
    prompt = (
        "Rewrite the query to be clearer and more specific. "
        "If it is already clear, return it unchanged. "
        "Keep it short and include key terms only."
    )
    try:
        rewritten = chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            num_predict=64,
        ).strip()
        return rewritten if rewritten else query
    except Exception:
        return query


def _file_signature(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _load_state() -> Dict[str, object]:
    if not STATE_PATH.exists():
        return {"files": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"files": {}}


def _save_state(state: Dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _detect_changes(data_dir: Path) -> Tuple[bool, Dict[str, object]]:
    state = _load_state()
    current_files: Dict[str, object] = {}
    for path in sorted(data_dir.glob("*.md")):
        current_files[str(path)] = _file_signature(path)
    changed = current_files != state.get("files", {})
    return changed, {"files": current_files}


def load_documents_from_markdown(
    data_dir: Path,
    max_tokens: int = 400,
    overlap_ratio: float = 0.1,
) -> List[Dict[str, object]]:
    if not data_dir.exists():
        if DEBUG:
            print(f"Data directory not found: {data_dir}")
        return []

    documents: List[Dict[str, object]] = []
    doc_id = 1
    for md_file in sorted(data_dir.glob("*.md")):
        chunks = chunk_markdown(
            md_file, max_tokens=max_tokens, overlap_ratio=overlap_ratio, debug=False
        )
        for chunk in chunks:
            meta = chunk["metadata"]
            documents.append(
                {
                    "id": doc_id,
                    "text": chunk["text"],
                    "metadata": {
                        "file_path": meta.get("file_path"),
                        "chunk_index": meta.get("chunk_index"),
                        "section": meta.get("section"),
                        "tokens": meta.get("tokens"),
                        "char_length": meta.get("char_length"),
                    },
                }
            )
            doc_id += 1

    if DEBUG:
        print(f"Loaded {len(documents)} chunks from {data_dir}")
    return documents


def initialize_pipeline() -> Dict[str, Any]:
    if DEBUG:
        print("Checking setup...")

    changed, state = _detect_changes(DATA_DIR)
    documents = load_documents_from_markdown(DATA_DIR)

    if not documents:
        raise RuntimeError("No markdown documents found. Aborting.")

    if DEBUG:
        print("Loading models...")

    client = QdrantClient(url="http://localhost:6333")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    if changed or not client.collection_exists("RagDocs"):
        if DEBUG:
            print("Changes detected in data folder. Rebuilding index...")
        index_documents(client, "RagDocs", embed_model, documents)
        _save_state(state)
    else:
        if DEBUG:
            print("No data changes detected. Using existing index.")

    bm25, _ = build_bm25_index(documents)

    if DEBUG:
        print("Setup complete. Ready for chat.")

    return {
        "client": client,
        "embed_model": embed_model,
        "reranker": reranker,
        "bm25": bm25,
        "documents": documents,
    }


def rag_pipeline(query: str, resources: Dict[str, Any]) -> Dict[str, object]:
    rewritten_query = rewrite_query(query)

    retrieved = hybrid_retrieve(
        rewritten_query,
        k=20,
        client=resources["client"],
        collection_name="RagDocs",
        embed_model=resources["embed_model"],
        bm25=resources["bm25"],
        documents=resources["documents"],
    )

    if DEBUG:
        print("Rewritten query:", rewritten_query)
        print("\nTop retrieved (pre-rerank):")
        for doc in retrieved[:5]:
            print(f"- score={doc.score:.4f} | {doc.metadata.get('file_path')}")
            print(f"  {doc.text[:120]}...")

    reranked = rerank(rewritten_query, retrieved[:20], k=5, reranker=resources["reranker"])

    if DEBUG:
        print("\nTop reranked:")
        for doc in reranked:
            print(f"- score={doc.score:.4f} | {doc.metadata.get('file_path')}")
            print(f"  {doc.text[:120]}...")

    context = filter_context(reranked, max_tokens=1500)
    prompt = build_prompt(context, rewritten_query)

    if DEBUG:
        print("\nFinal context:\n", context)
        print("\nFinal prompt:\n", prompt)

    answer = generate_answer(prompt)

    return {
        "query": query,
        "rewritten_query": rewritten_query,
        "documents": reranked,
        "answer": answer,
    }


if __name__ == "__main__":
    try:
        pipeline_resources = initialize_pipeline()
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(1)

    while True:
        user_query = input("Query: ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            break
        result = rag_pipeline(user_query, pipeline_resources)
        print("\nAnswer:\n", result["answer"])
