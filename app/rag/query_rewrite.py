from app.llm.client import chat


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
