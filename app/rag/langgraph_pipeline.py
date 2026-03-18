from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.answer_agent import answer_direct, answer_with_context
from app.agents.retrieval_agent import run_retrieval
from app.agents.router_agent import route_query
from app.rag.pipeline import initialize_pipeline
from app.utils.config import DEBUG, LANGSMITH_PROJECT, LOG_TRACE_RETRIEVAL
from app.utils.langsmith_logger import LangSmithLogger
from app.utils.logging import debug, info


class RagState(TypedDict):
    query: str
    route: str
    route_reason: str
    retrieval: Dict[str, Any]
    answer: Dict[str, Any]


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
    logger = LangSmithLogger(project=LANGSMITH_PROJECT)
    state: RagState = {
        "query": query,
        "route": "",
        "route_reason": "",
        "retrieval": {},
        "answer": {},
    }
    app = build_graph(resources)
    result = app.invoke(state)

    if DEBUG:
        info(f"Route: {result['route']} ({result['route_reason']})")
        if LOG_TRACE_RETRIEVAL and result.get("retrieval"):
            debug("Dense Top", items=result["retrieval"].get("trace", {}).get("dense_top", []))
            debug("BM25 Top", items=result["retrieval"].get("trace", {}).get("bm25_top", []))
            debug("Hybrid Top", items=result["retrieval"].get("trace", {}).get("hybrid_top", []))

    logger.log_event("router", {"route": result["route"], "reason": result["route_reason"]})
    logger.log_event(
        "retrieval",
        {
            "retrieved": len(result.get("retrieval", {}).get("retrieved", [])),
            "reranked": len(result.get("retrieval", {}).get("reranked", [])),
            "trace": result.get("retrieval", {}).get("trace", {}),
        },
    )
    logger.log_event("answer", {"sources": result.get("answer", {}).get("sources", [])})
    return result


def main() -> None:
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
