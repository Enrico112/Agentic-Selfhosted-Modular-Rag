from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from app.rag.pipeline import initialize_pipeline, run_query
import os
from pathlib import Path

app = FastAPI(title="RAG API", description="API for the RAG pipeline")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    route: str
    route_reason: str
    retrieval: dict
    answer: dict

# Global resources
resources = None

@app.on_event("startup")
async def startup_event():
    global resources
    try:
        resources = initialize_pipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        raise

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if resources is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    try:
        result = run_query(request.query, resources)
        return QueryResponse(
            route=result["route"],
            route_reason=result["route_reason"],
            retrieval=result["retrieval"],
            answer=result["answer"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/files")
async def list_data_files():
    data_dir = Path("data/goodwiki_markdown")
    if not data_dir.exists():
        return {"files": []}
    files = [f.name for f in data_dir.iterdir() if f.is_file()]
    return {"files": files}

@app.get("/api/settings")
async def get_settings():
    # Return current settings
    return {
        "embed_model": os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
        "reranker_model": os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base"),
        "llm_model": os.getenv("LLM_MODEL", "qwen2.5:7b"),
        "data_dir": os.getenv("DATA_DIR", "data/goodwiki_markdown"),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)