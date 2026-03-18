from __future__ import annotations

from typing import Any, Dict, List

from transformers import AutoTokenizer

from app.llm.client import generate_answer
from app.rag.prompt_manager import build_prompt
from app.retrieval.hybrid import Document
from app.utils.config import CONTEXT_MAX_TOKENS

_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    return _TOKENIZER


def _truncate_context(context: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not context:
        return context
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return context
    return tokenizer.decode(token_ids[:max_tokens])


def _collect_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    sources = []
    for doc in docs:
        sources.append(
            {
                "file_path": doc.metadata.get("file_path", "unknown"),
                "score": doc.score,
            }
        )
    return sources


def answer_with_context(query: str, context: str, docs: List[Document]) -> Dict[str, Any]:
    context = _truncate_context(context, CONTEXT_MAX_TOKENS)
    prompt = build_prompt(context, query)
    answer = generate_answer(prompt)
    return {
        "answer": answer,
        "sources": _collect_sources(docs),
        "prompt": prompt,
    }


def answer_direct(query: str) -> Dict[str, Any]:
    prompt = (
        "You are a helpful assistant. Answer directly and concisely.\n\n"
        f"Question:\n{query}"
    )
    answer = generate_answer(prompt)
    return {"answer": answer, "sources": [], "prompt": prompt}
