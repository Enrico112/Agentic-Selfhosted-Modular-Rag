from __future__ import annotations

from typing import Dict, List, Tuple

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


def _top_k_scores(
    score_map: Dict[int, float],
    payloads: Dict[int, Dict[str, object]] | None,
    k: int,
) -> List[Dict[str, object]]:
    sorted_ids = sorted(score_map, key=score_map.get, reverse=True)[:k]
    results: List[Dict[str, object]] = []
    for doc_id in sorted_ids:
        payload = payloads.get(doc_id, {}) if payloads else {}
        metadata = payload.get("metadata", {})
        results.append(
            {
                "id": doc_id,
                "score": score_map.get(doc_id, 0.0),
                "file_path": metadata.get("file_path", "unknown"),
            }
        )
    return results


def hybrid_retrieve_with_trace(
    query: str,
    k: int = 20,
    *,
    client: QdrantClient,
    collection_name: str,
    embed_model: SentenceTransformer,
    bm25: BM25Okapi,
    documents: List[Dict[str, object]],
    dense_alpha: float = 0.7,
) -> Tuple[List[Document], Dict[str, List[Dict[str, object]]]]:
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
        dense_weight = max(0.0, min(1.0, dense_alpha))
        sparse_weight = 1.0 - dense_weight
        fused_score = dense_weight * dense_norm.get(doc_id, 0.0) + sparse_weight * bm25_norm.get(doc_id, 0.0)
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
    trace = {
        "dense_top": _top_k_scores(dense_scores, dense_payloads, min(5, k)),
        "bm25_top": _top_k_scores(bm25_score_map, None, min(5, k)),
        "hybrid_top": [
            {
                "score": doc.score,
                "file_path": doc.metadata.get("file_path", "unknown"),
            }
            for doc in fused_docs[: min(5, k)]
        ],
    }
    return fused_docs[:k], trace


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
    docs, _ = hybrid_retrieve_with_trace(
        query,
        k,
        client=client,
        collection_name=collection_name,
        embed_model=embed_model,
        bm25=bm25,
        documents=documents,
    )
    return docs


def filter_context(
    docs: List[Document],
    max_tokens: int = 1500,
    *,
    max_docs: int = 5,
    deduplicate_by_file: bool = True,
) -> str:
    if not docs:
        return ""

    max_score = max(doc.score for doc in docs)
    score_threshold = max_score * 0.2 if max_score > 0 else 0.0

    seen = set()
    seen_files = set()
    chunks: List[str] = []
    token_count = 0

    for doc in docs:
        if doc.score < score_threshold:
            continue
        text = doc.text.strip()
        if not text or text in seen:
            continue
        source = str(doc.metadata.get("file_path", "unknown"))
        if deduplicate_by_file and source in seen_files:
            continue

        seen.add(text)
        seen_files.add(source)
        chunk = f"[Source: {source}]\n{text}"
        chunk_tokens = int(len(chunk.split()) * 1.3)
        if token_count + chunk_tokens > max_tokens:
            break
        chunks.append(chunk)
        token_count += chunk_tokens
        if len(chunks) >= max_docs:
            break

    return "\n\n".join(chunks)
