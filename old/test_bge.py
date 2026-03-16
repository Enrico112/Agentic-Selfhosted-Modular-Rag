# test_bge.py
import os

# Must be set before any HF import. FORCE safetensors usage to avoid PyTorch CVE issue ---
os.environ["HF_HUB_USE_SAFETENSORS"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings


# --- Initialize embeddings ---
embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",           # BGE-M3 model
    model_kwargs={"device": "cuda"},     # GPU usage
    encode_kwargs={"normalize_embeddings": True}
)

# --- Test texts ---
texts = [
    "Hello world",
    "What is agentic RAG?"
]

# --- Generate embeddings ---
vectors = embedder.embed_documents(texts)

# --- Output ---
print(f"Number of embeddings: {len(vectors)}")
print(f"Embedding dimension: {len(vectors[0])}")
print("Sample vector (first 5 values):", vectors[0][:5])