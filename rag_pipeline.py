from typing import Dict, List

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from llm import chat, generate_answer
from prompts import build_prompt
from retrieval import (
    Document,
    build_bm25_index,
    filter_context,
    hybrid_retrieve,
    index_documents,
    rerank,
)

DEBUG = True


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


def load_documents() -> List[Dict[str, object]]:
    return [
        {
            "id": 1,
            "text": (
                "Arden Holt is the VP of Product at Asteria Labs. "
                "She approved the Kitebeam roadmap on May 12, 2025 "
                "with a target launch date of September 3, 2025."
            ),
            "metadata": {"file_path": "docs/asteria_memo.md"},
        },
        {
            "id": 2,
            "text": (
                "BlueHaven Capital is a fictional venture firm founded in 2011 by Priya Arvan. "
                "Its flagship fund is Tidal II with a size of $420M."
            ),
            "metadata": {"file_path": "docs/bluehaven_snapshot.md"},
        },
        {
            "id": 3,
            "text": (
                "Novaline Health signed a 3-year contract on February 2, 2026 "
                "for the PulseFrame analytics platform at $3.2M annually."
            ),
            "metadata": {"file_path": "docs/novaline_contract.md"},
        },
        {
            "id": 4,
            "text": (
                "Nimbus Freight operates the Orchid logistics network. "
                "Its main hub is in Tallinn and the backup hub is in Riga."
            ),
            "metadata": {"file_path": "docs/nimbus_ops.md"},
        },
        {
            "id": 5,
            "text": (
                "SableWorks hired Dana Kirov as Head of People on August 8, 2024. "
                "The internal HR system codename is Glassleaf."
            ),
            "metadata": {"file_path": "docs/sableworks_hr.md"},
        },
    ]


def rag_pipeline(query: str) -> Dict[str, object]:
    rewritten_query = rewrite_query(query)

    documents = load_documents()
    client = QdrantClient(url="http://localhost:6333")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    index_documents(client, "RagDocs", embed_model, documents)
    bm25, _ = build_bm25_index(documents)

    retrieved = hybrid_retrieve(
        rewritten_query,
        k=20,
        client=client,
        collection_name="RagDocs",
        embed_model=embed_model,
        bm25=bm25,
        documents=documents,
    )

    if DEBUG:
        print("Rewritten query:", rewritten_query)
        print("\nTop retrieved (pre-rerank):")
        for doc in retrieved[:5]:
            print(f"- score={doc.score:.4f} | {doc.metadata.get('file_path')}")
            print(f"  {doc.text[:120]}...")

    reranked = rerank(rewritten_query, retrieved[:20], k=5, reranker=reranker)

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
    user_query = input("Query: ").strip()
    if user_query:
        result = rag_pipeline(user_query)
        print("\nAnswer:\n", result["answer"])
