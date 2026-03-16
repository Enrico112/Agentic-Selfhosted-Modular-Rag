import importlib.util
import subprocess
import sys

# 1. Install required packages if missing
def ensure_packages_installed() -> None:
    required = {
        "qdrant-client": "qdrant_client",
        "sentence-transformers": "sentence_transformers",
        "numpy": "numpy",
        "pandas": "pandas",
    }
    missing = [pkg for pkg, module in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_packages_installed()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# 2. Connect to local Qdrant instance
client = QdrantClient(url="http://localhost:6333")

# 3. Load embedding model (client-side embeddings for Qdrant)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
embedding_dim = model.get_sentence_embedding_dimension()

collection_name = "Document"

# Recreate collection to keep the example deterministic
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
)

# 4. Insert sample documents with metadata
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

# 5. Hybrid search with alpha=0.5 and top 5 results
query_text = "semantic keyword search with embeddings"
query_vector = model.encode(query_text, normalize_embeddings=True)
results = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=5,
    with_payload=True,
)

# 6. Print search results
for idx, point in enumerate(results, start=1):
    props = point.payload or {}
    print(f"Result {idx}")
    print("Title:", props.get("title"))
    print("Content:", props.get("content"))
    print("Tags:", props.get("tags"))
    print("-" * 40)
