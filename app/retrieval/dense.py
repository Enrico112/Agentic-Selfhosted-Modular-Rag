from __future__ import annotations

from typing import Dict, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.utils.logging import log


def index_documents(
    client: QdrantClient,
    collection_name: str,
    embed_model: SentenceTransformer,
    documents: List[Dict[str, object]],
    *,
    batch_size: int = 256,
) -> None:
    embedding_dim = embed_model.get_sentence_embedding_dimension()
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    total = len(documents)
    if total == 0:
        return

    if batch_size <= 0:
        batch_size = 256

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_docs = documents[start:end]
        texts = [str(doc["text"]) for doc in batch_docs]
        vectors = embed_model.encode(texts, normalize_embeddings=True)

        points = []
        for doc, vector in zip(batch_docs, vectors):
            payload = {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
            }
            points.append(PointStruct(id=doc["id"], vector=vector, payload=payload))

        client.upsert(collection_name=collection_name, points=points)
        log(f"Upserted {end}/{total} points")


def dense_search(
    client: QdrantClient,
    collection_name: str,
    embed_model: SentenceTransformer,
    query: str,
    k: int,
) -> Tuple[Dict[int, float], Dict[int, Dict[str, object]]]:
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

    return dense_scores, dense_payloads
