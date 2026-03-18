from app.llm.client import chat
from app.utils.config import ENABLE_INTENT_DETECTION, ENABLE_QUERY_REWRITE


def _detect_intent(query: str) -> str:
    lowered = query.lower()
    if any(word in lowered for word in ["summarize", "summary", "overview"]):
        return "summarize"
    if any(word in lowered for word in ["compare", "versus", "vs"]):
        return "compare"
    if any(word in lowered for word in ["extract", "list", "facts", "key facts"]):
        return "extract_facts"
    return "general"


def rewrite_query(query: str) -> str:
    if not ENABLE_QUERY_REWRITE:
        return query

    intent = _detect_intent(query) if ENABLE_INTENT_DETECTION else "general"
    prompt = (
        "Rewrite the query to be clearer and more specific for retrieval. "
        "If it is already clear, return it unchanged. "
        "Keep it short and include key terms only."
    )
    if intent == "summarize":
        prompt += " Focus on the subject and include 'summary' or 'overview' intent."
    elif intent == "compare":
        prompt += " Emphasize the items being compared and include both entities."
    elif intent == "extract_facts":
        prompt += " Emphasize factual extraction and include key entities or dates."
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
