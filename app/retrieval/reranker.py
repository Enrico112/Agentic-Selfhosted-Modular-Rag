from __future__ import annotations

from typing import List

from sentence_transformers import CrossEncoder

from app.retrieval.hybrid import Document


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
