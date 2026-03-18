from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


class Document:
    def __init__(self, text: str, score: float, metadata: Dict[str, object]):
        self.text = text
        self.score = score
        self.metadata = metadata


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]


def build_bm25_index(
    documents: List[Dict[str, object]],
) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized = [_tokenize(str(doc["text"])) for doc in documents]
    return BM25Okapi(tokenized), tokenized


def index_documents(
    client: QdrantClient,
    collection_name: str,
    embed_model: SentenceTransformer,
    documents: List[Dict[str, object]],
) -> None:
    embedding_dim = embed_model.get_sentence_embedding_dimension()
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    texts = [str(doc["text"]) for doc in documents]
    vectors = embed_model.encode(texts, normalize_embeddings=True)

    points = []
    for doc, vector in zip(documents, vectors):
        payload = {
            "id": doc["id"],
            "text": doc["text"],
            "metadata": doc.get("metadata", {}),
        }
        points.append(PointStruct(id=doc["id"], vector=vector, payload=payload))

    client.upsert(collection_name=collection_name, points=points)


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
    query_vector = embed_model.encode(query, normalize_embeddings=True)
    dense_results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k,
        with_payload=True,
    )

    dense_scores: Dict[int, float] = {}
    dense_payloads: Dict[int, Dict[str, object]] = {}
    for point in dense_results.points:
        payload = point.payload or {}
        doc_id = int(payload.get("id", point.id))
        dense_scores[doc_id] = float(point.score or 0.0)
        dense_payloads[doc_id] = payload

    bm25_scores_list = bm25.get_scores(_tokenize(query))
    bm25_scores = {int(doc["id"]): float(score) for doc, score in zip(documents, bm25_scores_list)}

    top_bm25_ids = sorted(bm25_scores, key=bm25_scores.get, reverse=True)[:k]
    candidate_ids = set(dense_scores.keys()) | set(top_bm25_ids)

    dense_norm = _normalize_scores({doc_id: dense_scores.get(doc_id, 0.0) for doc_id in candidate_ids})
    bm25_norm = _normalize_scores({doc_id: bm25_scores.get(doc_id, 0.0) for doc_id in candidate_ids})

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


def rerank(
    query: str,
    docs: List[Document],
    k: int = 5,
    *,
    reranker: CrossEncoder,
) -> List[Document]:
    if not docs:
        return []
    pairs = [(query, doc.text) for doc in docs]
    scores = reranker.predict(pairs)
    reranked: List[Document] = []
    for doc, score in zip(docs, scores):
        reranked.append(Document(text=doc.text, score=float(score), metadata=doc.metadata))
    reranked.sort(key=lambda d: d.score, reverse=True)
    return reranked[:k]


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
