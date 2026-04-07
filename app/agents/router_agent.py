from __future__ import annotations

from typing import Dict


def route_query(query: str) -> Dict[str, str]:
    lowered = query.lower()
    if any(word in lowered for word in ["summarize", "summary", "overview"]):
        return {"route": "summarize", "reason": "summary intent detected"}
    if any(word in lowered for word in ["compare", "versus", "vs"]):
        return {"route": "rag", "reason": "comparison requires grounded context"}
    if any(word in lowered for word in ["extract", "list", "facts", "key facts"]):
        return {"route": "rag", "reason": "fact extraction requires sources"}
    if any(word in lowered for word in ["who", "what", "when", "where", "why", "how"]):
        return {"route": "rag", "reason": "question likely benefits from retrieval"}
    return {"route": "direct", "reason": "general chat"}
