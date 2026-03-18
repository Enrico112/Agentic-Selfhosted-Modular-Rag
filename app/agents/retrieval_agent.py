from __future__ import annotations

from typing import Any, Dict, List

from app.retrieval.hybrid import Document, filter_context, hybrid_retrieve_with_trace
from app.retrieval.reranker import rerank
from app.utils.config import (
    CONTEXT_MAX_DOCS,
    CONTEXT_MAX_TOKENS,
    DEDUPLICATE_BY_FILE,
    DENSE_ALPHA,
    RERANK_K,
    RETRIEVE_K,
)


def run_retrieval(query: str, resources: Dict[str, Any]) -> Dict[str, Any]:
    retrieved, trace = hybrid_retrieve_with_trace(
        query,
        k=RETRIEVE_K,
        client=resources["client"],
        collection_name="RagDocs",
        embed_model=resources["embed_model"],
        bm25=resources["bm25"],
        documents=resources["documents"],
        dense_alpha=DENSE_ALPHA,
    )

    reranked = rerank(
        query,
        retrieved[:RETRIEVE_K],
        k=RERANK_K,
        reranker=resources["reranker"],
    )

    context = filter_context(
        reranked,
        max_tokens=CONTEXT_MAX_TOKENS,
        max_docs=CONTEXT_MAX_DOCS,
        deduplicate_by_file=DEDUPLICATE_BY_FILE,
    )

    return {
        "retrieved": retrieved,
        "reranked": reranked,
        "context": context,
        "trace": trace,
    }
