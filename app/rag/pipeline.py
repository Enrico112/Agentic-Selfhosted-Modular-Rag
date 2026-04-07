from __future__ import annotations

from typing import Any, Dict, List, TypedDict
from datetime import datetime, timezone
from pathlib import Path
import json

from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer
from dotenv import load_dotenv

from app.agents.answer_agent import answer_direct, answer_with_context
from app.agents.retrieval_agent import run_retrieval
from app.agents.router_agent import route_query
from app.ingestion.watcher import (
    load_documents_from_markdown,
    should_reindex,
    commit_state,
)
from app.retrieval.dense import index_documents
from app.retrieval.sparse import build_bm25_index
from app.utils.config import (
    DATA_DIR,
    STATE_PATH,
    LANGGRAPH_USE_LANGSMITH_API,
    LOCAL_TRACE_PATH,
    LOG_LEVEL,
    LOG_TRACE_RETRIEVAL,
    EMBED_MODEL_NAME,
    RERANKER_MODEL_NAME,
)
from app.utils.langsmith_logger import configure_langsmith_tracing
from app.utils.logging import debug, info

DATA_DIR = Path(DATA_DIR)
STATE_PATH = Path(STATE_PATH)


class RagState(TypedDict):
    query: str
    route: str
    route_reason: str
    retrieval: Dict[str, Any]
    answer: Dict[str, Any]


def initialize_pipeline() -> Dict[str, Any]:
    info("Checking setup...")

    changed, state = should_reindex(DATA_DIR, STATE_PATH)
    documents = load_documents_from_markdown(DATA_DIR)

    if not documents:
        raise RuntimeError("No markdown documents found. Aborting.")

    info("Loading models...")

    client = QdrantClient(url="http://localhost:6333")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    if changed or not client.collection_exists("RagDocs"):
        info("Changes detected in data folder. Rebuilding index...")
        index_documents(client, "RagDocs", embed_model, documents)
        commit_state(STATE_PATH, state)
    else:
        info("No data changes detected. Using existing index.")

    bm25, _ = build_bm25_index(documents)

    configure_langsmith_tracing(LANGGRAPH_USE_LANGSMITH_API)
    info("Setup complete. Ready for chat.")

    return {
        "client": client,
        "embed_model": embed_model,
        "reranker": reranker,
        "bm25": bm25,
        "documents": documents,
    }


def _router_node(state: RagState) -> RagState:
    decision = route_query(state["query"])
    state["route"] = decision["route"]
    state["route_reason"] = decision["reason"]
    return state


def _retrieval_node(state: RagState, resources: Dict[str, Any]) -> RagState:
    if state["route"] in {"rag", "summarize"}:
        retrieval = run_retrieval(state["query"], resources)
    else:
        retrieval = {"retrieved": [], "reranked": [], "context": "", "trace": {}}
    state["retrieval"] = retrieval
    return state


def _answer_node(state: RagState) -> RagState:
    if state["route"] in {"rag", "summarize"}:
        answer = answer_with_context(
            query=state["query"],
            context=state["retrieval"]["context"],
            docs=state["retrieval"]["reranked"],
        )
    else:
        answer = answer_direct(state["query"])
    state["answer"] = answer
    return state


def build_graph(resources: Dict[str, Any]):
    graph = StateGraph(RagState)
    graph.add_node("router", _router_node)
    graph.add_node("retrieval", lambda state: _retrieval_node(state, resources))
    graph.add_node("answer", _answer_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "retrieval")
    graph.add_edge("retrieval", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def run_query(query: str, resources: Dict[str, Any]) -> Dict[str, Any]:
    state: RagState = {
        "query": query,
        "route": "",
        "route_reason": "",
        "retrieval": {},
        "answer": {},
    }
    app = build_graph(resources)
    result = app.invoke(state)

    if LOG_LEVEL == "DEBUG":
        info(f"Route: {result['route']} ({result['route_reason']})")
        if LOG_TRACE_RETRIEVAL and result.get("retrieval"):
            debug("Dense Top", items=result["retrieval"].get("trace", {}).get("dense_top", []))
            debug("BM25 Top", items=result["retrieval"].get("trace", {}).get("bm25_top", []))
            debug("Hybrid Top", items=result["retrieval"].get("trace", {}).get("hybrid_top", []))

    if not LANGGRAPH_USE_LANGSMITH_API:
        _write_local_trace(query, result)
    return result


def _write_local_trace(query: str, result: Dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "route": result.get("route"),
        "route_reason": result.get("route_reason"),
        "retrieved_count": len(result.get("retrieval", {}).get("retrieved", []) or []),
        "reranked_count": len(result.get("retrieval", {}).get("reranked", []) or []),
        "sources": result.get("answer", {}).get("sources", []),
        "answer": result.get("answer", {}).get("answer", ""),
    }
    path = Path(LOCAL_TRACE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv()
    resources = initialize_pipeline()
    while True:
        query = input("Query: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        result = run_query(query, resources)
        print("\nAnswer:\n", result["answer"]["answer"])


if __name__ == "__main__":
    main()
