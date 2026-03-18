from __future__ import annotations

from typing import Dict, List

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.retrieval.dense import dense_search
from app.retrieval.sparse import bm25_scores

class Document:
    def __init__(self, text: str, score: float, metadata: Dict[str, object]):
        self.text = text
        self.score = score
        self.metadata = metadata


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return {doc_id: 0.0 for doc_id in scores}
    return {doc_id: (score - min_val) / (max_val - min_val) for doc_id, score in scores.items()}


def hybrid_retrieve(
    query: str,
    k: int = 20,
    *,
    client: QdrantClient,
    collection_name: str,
    embed_model: SentenceTransformer,
    bm25: BM25Okapi,
    documents: List[Dict[str, object]],
) -> List[Document]:
    dense_scores, dense_payloads = dense_search(
        client,
        collection_name,
        embed_model,
        query,
        k,
    )

    bm25_score_map = bm25_scores(bm25, query, documents)

    top_bm25_ids = sorted(bm25_score_map, key=bm25_score_map.get, reverse=True)[:k]
    candidate_ids = set(dense_scores.keys()) | set(top_bm25_ids)

    dense_norm = _normalize_scores({doc_id: dense_scores.get(doc_id, 0.0) for doc_id in candidate_ids})
    bm25_norm = _normalize_scores({doc_id: bm25_score_map.get(doc_id, 0.0) for doc_id in candidate_ids})

    doc_map = {int(doc["id"]): doc for doc in documents}
    fused_docs: List[Document] = []
    for doc_id in candidate_ids:
        fused_score = 0.7 * dense_norm.get(doc_id, 0.0) + 0.3 * bm25_norm.get(doc_id, 0.0)
        if doc_id in dense_payloads:
            payload = dense_payloads[doc_id]
            text = str(payload.get("text", ""))
            metadata = payload.get("metadata", {})
        else:
            doc = doc_map[doc_id]
            text = str(doc["text"])
            metadata = doc.get("metadata", {})
        fused_docs.append(Document(text=text, score=fused_score, metadata=metadata))

    fused_docs.sort(key=lambda d: d.score, reverse=True)
    return fused_docs[:k]


def filter_context(docs: List[Document], max_tokens: int = 1500) -> str:
    if not docs:
        return ""

    max_score = max(doc.score for doc in docs)
    score_threshold = max_score * 0.2 if max_score > 0 else 0.0

    seen = set()
    chunks: List[str] = []
    token_count = 0

    for doc in docs:
        if doc.score < score_threshold:
            continue
        text = doc.text.strip()
        if not text or text in seen:
            continue
        seen.add(text)

        source = str(doc.metadata.get("file_path", "unknown"))
        chunk = f"[Source: {source}]\n{text}"
        chunk_tokens = int(len(chunk.split()) * 1.3)
        if token_count + chunk_tokens > max_tokens:
            break
        chunks.append(chunk)
        token_count += chunk_tokens

    return "\n\n".join(chunks)
