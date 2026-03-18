from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Tuple

from transformers import AutoTokenizer

DEBUG = False
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    return _TOKENIZER


def _count_tokens(text: str) -> int:
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens])


def _tail_tokens(text: str, token_count: int) -> str:
    if token_count <= 0:
        return ""
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids[-token_count:])


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _clean_lines(lines: List[str]) -> List[str]:
    cleaned = [line.rstrip() for line in lines]
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return cleaned


def _parse_sections(markdown: str) -> List[Dict[str, object]]:
    header_re = re.compile(r"^(#{1,3})\s+(.*)$")
    lines = markdown.splitlines()

    sections: List[Dict[str, object]] = []
    current = {"header_line": None, "section_title": "Preamble", "content_lines": []}
    h1 = ""
    h2 = ""

    for line in lines:
        match = header_re.match(line)
        if match:
            if current["header_line"] or current["content_lines"]:
                sections.append(current)
            level = len(match.group(1))
            title = match.group(2).strip()

            if level == 1:
                h1 = title
                h2 = ""
                section_title = h1
            elif level == 2:
                h2 = title
                section_title = h2 or h1 or title
            else:
                section_title = h2 or h1 or title

            current = {
                "header_line": line.rstrip(),
                "section_title": section_title,
                "content_lines": [],
            }
        else:
            current["content_lines"].append(line)

    if current["header_line"] or current["content_lines"]:
        sections.append(current)

    return sections


def _section_chunks(
    header_line: str | None,
    content_lines: List[str],
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    content_lines = _clean_lines(content_lines)
    content = "\n".join(content_lines).strip()
    if not content and not header_line:
        return []

    if not content:
        header_only = header_line.strip() if header_line else ""
        return [header_only] if header_only else []

    paragraphs = _split_paragraphs(content)
    units: List[str] = []
    for paragraph in paragraphs:
        if _count_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
        else:
            sentences = _split_sentences(paragraph)
            units.extend(sentences if sentences else [paragraph])

    chunks: List[str] = []
    carry = ""

    def start_chunk() -> str:
        base = header_line.strip() if header_line else ""
        if base:
            base = base + "\n\n"
        if carry:
            return base + carry.strip()
        return base.rstrip()

    current = ""

    for unit in units:
        unit_text = unit.strip()
        if not current:
            current = start_chunk()

        separator = "\n\n" if current else ""
        candidate = f"{current}{separator}{unit_text}" if current else unit_text

        if _count_tokens(candidate) <= max_tokens:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            carry = _tail_tokens(current, overlap_tokens).strip()
            current = ""

        current = start_chunk()
        separator = "\n\n" if current else ""
        candidate = f"{current}{separator}{unit_text}" if current else unit_text

        if _count_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            remaining = max_tokens - _count_tokens(current)
            truncated = _truncate_to_tokens(unit_text, remaining)
            current = f"{current}{separator}{truncated}".strip()

    if current:
        chunks.append(current.strip())

    return chunks


def chunk_markdown(
    file_path: str | Path,
    max_tokens: int = 400,
    overlap_ratio: float = 0.1,
    debug: bool | None = None,
) -> List[Dict[str, object]]:
    file_path = Path(file_path)
    markdown = file_path.read_text(encoding="utf-8")
    sections = _parse_sections(markdown)

    overlap_tokens = max(0, int(max_tokens * overlap_ratio))

    chunks: List[Dict[str, object]] = []
    chunk_index = 0

    for section in sections:
        section_chunks = _section_chunks(
            header_line=section["header_line"],
            content_lines=section["content_lines"],
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        for text in section_chunks:
            tokens = _count_tokens(text)
            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        "file_path": str(file_path),
                        "chunk_index": chunk_index,
                        "section": section["section_title"],
                        "tokens": tokens,
                        "char_length": len(text),
                    },
                }
            )
            chunk_index += 1

    if (debug if debug is not None else DEBUG):
        print(f"[chunk_markdown] {file_path}: {len(chunks)} chunks")
        for chunk in chunks:
            meta = chunk["metadata"]
            print(
                f"  - idx={meta['chunk_index']} tokens={meta['tokens']} "
                f"section={meta['section']}"
            )

    return chunks
