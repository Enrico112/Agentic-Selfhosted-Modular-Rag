from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from app.ingestion.markdown_chunker import chunk_markdown
from app.utils.config import DEBUG
from app.utils.file_utils import detect_changes, save_state
from app.utils.logging import log


def load_documents_from_markdown(
    data_dir: Path,
    max_tokens: int = 400,
    overlap_ratio: float = 0.1,
) -> List[Dict[str, object]]:
    if not data_dir.exists():
        log(f"Data directory not found: {data_dir}")
        return []

    documents: List[Dict[str, object]] = []
    doc_id = 1
    for md_file in sorted(data_dir.glob("*.md")):
        chunks = chunk_markdown(
            md_file, max_tokens=max_tokens, overlap_ratio=overlap_ratio, debug=False
        )
        for chunk in chunks:
            meta = chunk["metadata"]
            documents.append(
                {
                    "id": doc_id,
                    "text": chunk["text"],
                    "metadata": {
                        "file_path": meta.get("file_path"),
                        "chunk_index": meta.get("chunk_index"),
                        "section": meta.get("section"),
                        "tokens": meta.get("tokens"),
                        "char_length": meta.get("char_length"),
                    },
                }
            )
            doc_id += 1

    log(f"Loaded {len(documents)} chunks from {data_dir}")
    return documents


def should_reindex(data_dir: Path, state_path: Path) -> Tuple[bool, Dict[str, object]]:
    return detect_changes(data_dir, state_path)


def commit_state(state_path: Path, state: Dict[str, object]) -> None:
    save_state(state_path, state)
