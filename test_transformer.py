import importlib.util
import subprocess
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Connect to local Qdrant instance
client = QdrantClient(url="http://localhost:6333")

# Load embedding model (client-side embeddings for Qdrant)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
embedding_dim = model.get_sentence_embedding_dimension()

collection_name = "Document"

# Recreate collection to keep the example deterministic
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
)

# Insert sample documents with metadata
documents = [
    {
        "title": "Hybrid Search Basics",
        "content": "Hybrid search blends keyword matching with semantic vector search.",
        "tags": ["search", "hybrid", "overview"],
    },
    {
        "title": "Qdrant and Transformers",
        "content": "Qdrant stores vectors created by transformer models for fast similarity search.",
        "tags": ["qdrant", "transformers", "embeddings"],
    },
    {
        "title": "RAG Fundamentals",
        "content": "Retrieval-augmented generation uses a retriever to ground LLM outputs.",
        "tags": ["rag", "llm", "retrieval"],
    },
    {
        "title": "Keyword Search Tips",
        "content": "Keyword search is fast and precise but can miss semantic similarity.",
        "tags": ["keyword", "search", "precision"],
    },
    {
        "title": "Vector Search Essentials",
        "content": "Vector search finds semantically related content using embeddings.",
        "tags": ["vector", "search", "semantics"],
    },
]

texts = [f"{d['title']}\n{d['content']}" for d in documents]
vectors = model.encode(texts, normalize_embeddings=True)

points = [
    PointStruct(id=idx, vector=vector, payload=doc)
    for idx, (vector, doc) in enumerate(zip(vectors, documents), start=1)
]

client.upsert(collection_name=collection_name, points=points)

# Hybrid search with alpha=0.5 and top 5 results
query_text = "semantic keyword search with embeddings"
query_vector = model.encode(query_text, normalize_embeddings=True)
results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=5,
    with_payload=True,
)

# Print search results
for idx, point in enumerate(results.points, start=1):
    props = point.payload or {}
    print(f"Result {idx}")
    print("Title:", props.get("title"))
    print("Content:", props.get("content"))
    print("Tags:", props.get("tags"))
    print("-" * 40)
