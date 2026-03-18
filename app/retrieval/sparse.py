from __future__ import annotations

import re
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]


def build_bm25_index(
    documents: List[Dict[str, object]],
) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized = [_tokenize(str(doc["text"])) for doc in documents]
    return BM25Okapi(tokenized), tokenized


def bm25_scores(
    bm25: BM25Okapi,
    query: str,
    documents: List[Dict[str, object]],
) -> Dict[int, float]:
    scores = bm25.get_scores(_tokenize(query))
    return {int(doc["id"]): float(score) for doc, score in zip(documents, scores)}
