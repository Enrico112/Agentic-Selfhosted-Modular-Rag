from typing import Dict, Any, List
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer
import json
from transformers import AutoTokenizer
from dotenv import load_dotenv

from app.llm.client import generate_answer
from app.rag.prompt_manager import build_prompt
from app.rag.query_rewrite import rewrite_query
from app.ingestion.watcher import (
    load_documents_from_markdown,
    should_reindex,
    commit_state,
)
from app.retrieval.dense import index_documents
from app.retrieval.hybrid import Document, filter_context, hybrid_retrieve_with_trace
from app.retrieval.reranker import rerank
from app.retrieval.sparse import build_bm25_index
from app.utils.config import (
    DATA_DIR,
    STATE_PATH,
    DENSE_ALPHA,
    RETRIEVE_K,
    RERANK_K,
    TOPK_TRACE,
    CONTEXT_MAX_TOKENS,
    CONTEXT_MAX_DOCS,
    DEDUPLICATE_BY_FILE,
    COMPRESS_LOW_RELEVANCE,
    COMPRESS_THRESHOLD_RATIO,
    LOG_TRACE_RETRIEVAL,
    SAVE_QUERY_HISTORY,
    QUERY_HISTORY_PATH,
    LOG_LEVEL,
    LANGGRAPH_USE_LANGSMITH_API,
)
from app.utils.logging import info, debug
from app.utils.langsmith_logger import configure_langsmith_tracing

DATA_DIR = Path(DATA_DIR)
STATE_PATH = Path(STATE_PATH)
QUERY_HISTORY_PATH = Path(QUERY_HISTORY_PATH)

_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        # Only used for token counting/truncation, not model inference.
        _TOKENIZER.model_max_length = 100_000
    return _TOKENIZER


def _count_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_context(context: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not context:
        return context
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return context
    return tokenizer.decode(token_ids[:max_tokens])


def _persist_query_history(entry: Dict[str, Any]) -> None:
    if not SAVE_QUERY_HISTORY:
        return
    QUERY_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUERY_HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def initialize_pipeline() -> Dict[str, Any]:
    info("Checking setup...")

    changed, state = should_reindex(DATA_DIR, STATE_PATH)
    documents = load_documents_from_markdown(DATA_DIR)

    if not documents:
        raise RuntimeError("No markdown documents found. Aborting.")

    info("Loading models...")

    client = QdrantClient(url="http://localhost:6333")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    if changed or not client.collection_exists("RagDocs"):
        info("Changes detected in data folder. Rebuilding index...")
        index_documents(client, "RagDocs", embed_model, documents)
        commit_state(STATE_PATH, state)
    else:
        info("No data changes detected. Using existing index.")

    bm25, _ = build_bm25_index(documents)

    configure_langsmith_tracing(LANGGRAPH_USE_LANGSMITH_API)
    info("Setup complete. Ready for chat.")

    return {
        "client": client,
        "embed_model": embed_model,
        "reranker": reranker,
        "bm25": bm25,
        "documents": documents,
    }


def rag_pipeline(query: str, resources: Dict[str, Any]) -> Dict[str, object]:
    rewritten_query = rewrite_query(query)

    retrieved, trace = hybrid_retrieve_with_trace(
        rewritten_query,
        k=RETRIEVE_K,
        client=resources["client"],
        collection_name="RagDocs",
        embed_model=resources["embed_model"],
        bm25=resources["bm25"],
        documents=resources["documents"],
        dense_alpha=DENSE_ALPHA,
    )

    if LOG_LEVEL == "DEBUG":
        print("Rewritten query:", rewritten_query)
        if LOG_TRACE_RETRIEVAL:
            debug("Dense Top", items=trace.get("dense_top", []))
            debug("BM25 Top", items=trace.get("bm25_top", []))
            debug("Hybrid Top", items=trace.get("hybrid_top", []))
        print("\nTop retrieved (pre-rerank):")
        for doc in retrieved[: TOPK_TRACE]:
            print(f"- score={doc.score:.4f} | {doc.metadata.get('file_path')}")
            print(f"  {doc.text[:120]}...")

    reranked = rerank(rewritten_query, retrieved[:RETRIEVE_K], k=RERANK_K, reranker=resources["reranker"])

    if LOG_LEVEL == "DEBUG":
        print("\nTop reranked:")
        for doc in reranked[: TOPK_TRACE]:
            print(f"- score={doc.score:.4f} | {doc.metadata.get('file_path')}")
            print(f"  {doc.text[:120]}...")

    context_docs = reranked
    if COMPRESS_LOW_RELEVANCE and reranked:
        threshold = max(doc.score for doc in reranked) * COMPRESS_THRESHOLD_RATIO
        for idx, doc in enumerate(reranked):
            if doc.score >= threshold:
                continue
            summary_prompt = (
                "Summarize the following passage in 1-2 sentences, preserving key facts:\n\n"
                f"{doc.text}"
            )
            summary = generate_answer(summary_prompt)
            context_docs[idx] = Document(
                text=summary,
                score=doc.score,
                metadata=doc.metadata,
            )

    context = filter_context(
        context_docs,
        max_tokens=CONTEXT_MAX_TOKENS,
        max_docs=CONTEXT_MAX_DOCS,
        deduplicate_by_file=DEDUPLICATE_BY_FILE,
    )
    context = _truncate_context(context, CONTEXT_MAX_TOKENS)
    prompt = build_prompt(context, rewritten_query)
    prompt_tokens = _count_tokens(prompt)

    if LOG_LEVEL == "DEBUG":
        print("\nFinal context:\n", context)
        print("\nFinal prompt:\n", prompt)
        debug("Prompt stats", prompt_tokens=prompt_tokens)

    answer = generate_answer(prompt)

    _persist_query_history(
        {
            "query": query,
            "rewritten_query": rewritten_query,
            "retrieved_count": len(retrieved),
            "reranked_count": len(reranked),
            "prompt_tokens": prompt_tokens,
        }
    )

    return {
        "query": query,
        "rewritten_query": rewritten_query,
        "documents": reranked,
        "answer": answer,
    }


if __name__ == "__main__":
    try:
        load_dotenv()
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
