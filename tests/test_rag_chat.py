import sys
from typing import List, Dict, Tuple

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


def build_documents() -> List[Dict[str, object]]:
    return [
        {
            "title": "Arden Holt - Product Memo (Asteria Labs)",
            "content": (
                "Arden Holt is the VP of Product at Asteria Labs. She wrote the May 12, 2025 memo "
                "approving the 'Kitebeam' roadmap with a target launch of September 3, 2025. "
                "Her direct report is Milo Greaves, and her team sits on floor 14 in the Lisbon office."
            ),
            "tags": ["asteria", "arden-holt", "product", "memo"],
        },
        {
            "title": "BlueHaven Capital - Fund Snapshot",
            "content": (
                "BlueHaven Capital is a fictional venture firm founded in 2011 by Priya Arvan. "
                "Their flagship fund is 'Tidal II' with a size of $420M, and their office address "
                "is 77 Harbor Pike, Seattle. The current CTO is Ezra Lang."
            ),
            "tags": ["bluehaven", "venture", "fund", "snapshot"],
        },
        {
            "title": "Novaline Health - Customer Contract",
            "content": (
                "Novaline Health signed a 3-year contract on February 2, 2026 for its 'PulseFrame' "
                "analytics platform. The contract value is $3.2M annually, and the renewal window "
                "opens on October 15, 2028. Account owner: Linh Alvarez."
            ),
            "tags": ["novaline", "contract", "health", "pulseframe"],
        },
        {
            "title": "Nimbus Freight - Ops Note",
            "content": (
                "Nimbus Freight operates the 'Orchid' logistics network. Their main hub is in "
                "Tallinn, and the backup hub is in Riga. The COO, Jae Nakamoto, approved the "
                "Q4 routing change that reduced average delivery time to 38 hours."
            ),
            "tags": ["nimbus", "logistics", "ops", "orchid"],
        },
        {
            "title": "SableWorks - HR Profile",
            "content": (
                "SableWorks hired Dana Kirov as Head of People on August 8, 2024. "
                "Her previous role was at Finch & Rowe. The internal HR system codename is "
                "'Glassleaf' and the employee policy refresh is due March 30, 2026."
            ),
            "tags": ["sableworks", "hr", "people", "glassleaf"],
        },
    ]


def setup_qdrant(
    client: QdrantClient,
    collection_name: str,
    model: SentenceTransformer,
    documents: List[Dict[str, object]],
) -> None:
    embedding_dim = model.get_sentence_embedding_dimension()
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    texts = [f"{d['title']}\n{d['content']}" for d in documents]
    vectors = model.encode(texts, normalize_embeddings=True)

    points = [
        PointStruct(
            id=idx,
            vector=vector,
            payload={
                **doc,
                "text": text,
            },
        )
        for idx, (vector, doc, text) in enumerate(zip(vectors, documents, texts), start=1)
    ]
    client.upsert(collection_name=collection_name, points=points)


def keyword_overlap_score(query: str, text: str) -> float:
    query_terms = {t.lower() for t in query.split() if t.strip()}
    if not query_terms:
        return 0.0
    text_terms = {t.lower() for t in text.split() if t.strip()}
    return len(query_terms & text_terms) / len(query_terms)


def hybrid_rerank(
    query: str,
    scored_points: List[Tuple[float, Dict[str, object]]],
) -> List[Tuple[float, Dict[str, object]]]:
    reranked = []
    for vector_score, payload in scored_points:
        text = payload.get("text", "")
        keyword_score = keyword_overlap_score(query, text)
        # Simple hybrid: 70% vector, 30% keyword overlap
        combined = 0.7 * vector_score + 0.3 * keyword_score
        reranked.append((combined, payload))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked


def should_use_rag(top_score: float) -> bool:
    return top_score >= 0.25


def build_context(docs: List[Dict[str, object]], limit: int = 3) -> str:
    chunks = []
    for doc in docs[:limit]:
        chunks.append(f"Title: {doc.get('title')}\nContent: {doc.get('content')}")
    return "\n\n".join(chunks)


def main() -> None:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(model_name)

    client = QdrantClient(url="http://localhost:6333")
    collection_name = "RagDocuments"
    documents = build_documents()
    setup_qdrant(client, collection_name, embedding_model, documents)

    print("Type a question. Use 'exit' to quit.")
    while True:
        user_query = input("You: ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            break

        query_vector = embedding_model.encode(user_query, normalize_embeddings=True)
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=5,
            with_payload=True,
        )

        scored_points = []
        for point in results.points:
            payload = point.payload or {}
            scored_points.append((float(point.score or 0.0), payload))

        reranked = hybrid_rerank(user_query, scored_points)
        top_score = reranked[0][0] if reranked else 0.0

        if should_use_rag(top_score):
            context_docs = [payload for _, payload in reranked[:3]]
            context = build_context(context_docs)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Use the provided context when relevant. "
                        "If the context is not enough, say so and answer briefly anyway."
                    ),
                },
                {"role": "system", "content": f"Context:\n{context}"},
                {"role": "user", "content": user_query},
            ]
        else:
            messages = [{"role": "user", "content": user_query}]

        try:
            response = ollama.chat(model="qwen2.5:7b", messages=messages)
        except Exception as exc:
            print(f"Error calling Ollama: {exc}")
            sys.exit(1)

        print("\nAssistant:")
        print(response["message"]["content"])
        print()


if __name__ == "__main__":
    main()
