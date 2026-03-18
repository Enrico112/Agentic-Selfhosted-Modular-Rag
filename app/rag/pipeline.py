from typing import Dict, Any
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.llm.client import generate_answer
from app.rag.prompt_manager import build_prompt
from app.rag.query_rewrite import rewrite_query
from app.ingestion.watcher import (
    load_documents_from_markdown,
    should_reindex,
    commit_state,
)
from app.retrieval.dense import index_documents
from app.retrieval.hybrid import Document, filter_context, hybrid_retrieve
from app.retrieval.reranker import rerank
from app.retrieval.sparse import build_bm25_index
from app.utils.config import DEBUG
from app.utils.logging import log

DATA_DIR = Path("data/goodwiki_markdown_sample")
STATE_PATH = Path("data/.rag_index_state.json")


def initialize_pipeline() -> Dict[str, Any]:
    log("Checking setup...")

    changed, state = should_reindex(DATA_DIR, STATE_PATH)
    documents = load_documents_from_markdown(DATA_DIR)

    if not documents:
        raise RuntimeError("No markdown documents found. Aborting.")

    log("Loading models...")

    client = QdrantClient(url="http://localhost:6333")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    if changed or not client.collection_exists("RagDocs"):
        log("Changes detected in data folder. Rebuilding index...")
        index_documents(client, "RagDocs", embed_model, documents)
        commit_state(STATE_PATH, state)
    else:
        log("No data changes detected. Using existing index.")

    bm25, _ = build_bm25_index(documents)

    log("Setup complete. Ready for chat.")

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
