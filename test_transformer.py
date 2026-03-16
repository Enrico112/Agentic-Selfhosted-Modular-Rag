import importlib.util
import subprocess
import sys

# 1. Install required packages if missing
def ensure_packages_installed() -> None:
    required = {
        "weaviate-client": "weaviate",
        "sentence-transformers": "sentence_transformers",
        "numpy": "numpy",
        "pandas": "pandas",
    }
    missing = [pkg for pkg, module in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_packages_installed()

import weaviate
from weaviate.classes.config import Property, DataType, Configure

# 2. Connect to local Weaviate instance
client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=8080,
    http_secure=False,
)

# 3. Define schema for "Document" using text2vec-transformers
# Note: Weaviate generates embeddings internally via the module, so we do not embed here.
if client.collections.exists("Document"):
    client.collections.delete("Document")

client.collections.create(
    name="Document",
    vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
        Property(name="tags", data_type=DataType.TEXT_ARRAY),
    ],
)

collection = client.collections.get("Document")

# 4. Insert sample documents with metadata
documents = [
    {
        "title": "Hybrid Search Basics",
        "content": "Hybrid search blends keyword matching with semantic vector search.",
        "tags": ["search", "hybrid", "overview"],
    },
    {
        "title": "Weaviate and Transformers",
        "content": "Weaviate can use text2vec-transformers to vectorize content automatically.",
        "tags": ["weaviate", "transformers", "embeddings"],
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

for doc in documents:
    collection.data.insert(properties=doc)

# 5. Hybrid search with alpha=0.5 and top 5 results
query_text = "semantic keyword search with embeddings"
results = collection.query.hybrid(
    query=query_text,
    alpha=0.5,
    limit=5,
    return_properties=["title", "content", "tags"],
)

# 6. Print search results
for idx, obj in enumerate(results.objects, start=1):
    props = obj.properties
    print(f"Result {idx}")
    print("Title:", props.get("title"))
    print("Content:", props.get("content"))
    print("Tags:", props.get("tags"))
    print("-" * 40)

client.close()
