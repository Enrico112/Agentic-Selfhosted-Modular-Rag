def build_prompt(context: str, query: str) -> str:
    return (
        "You are a retrieval-augmented assistant.\n\n"
        "Rules:\n"
        "- Only answer using the provided context\n"
        "- If the answer is not in the context, say: \"I don't know\"\n"
        "- Do NOT make up information\n"
        "- Be concise and accurate\n"
        "- Cite sources using [file_path]\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}"
    )
