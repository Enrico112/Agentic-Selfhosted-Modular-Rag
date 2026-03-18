from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from app.llm.client import chat
from app.rag.langgraph_pipeline import run_query
from app.rag.pipeline import initialize_pipeline
from app.utils.config import DATA_DIR, LANGGRAPH_USE_LANGSMITH_API
from app.utils.logging import info


def _normalize_title(name: str) -> str:
    return name.replace("_", " ").strip()


def build_questions(data_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(data_dir.glob("*.md"))
    if len(files) < 10:
        raise RuntimeError(f"Need at least 10 markdown files in {data_dir}")
    titles = [_normalize_title(p.stem) for p in files[:10]]

    return [
        {"id": 1, "question": f"What is {titles[0]}?", "expected_route": "rag", "expected_unknown": False},
        {"id": 2, "question": f"What is {titles[1]}?", "expected_route": "rag", "expected_unknown": False},
        {"id": 3, "question": f"What is {titles[2]}?", "expected_route": "rag", "expected_unknown": False},
        {"id": 4, "question": f"What is {titles[3]}?", "expected_route": "rag", "expected_unknown": False},
        {"id": 5, "question": f"When did {titles[4]} take place?", "expected_route": "rag", "expected_unknown": False},
        {"id": 6, "question": f"When did {titles[5]} take place?", "expected_route": "rag", "expected_unknown": False},
        {"id": 7, "question": f"Summarize {titles[6]} in two sentences.", "expected_route": "summarize", "expected_unknown": False},
        {
            "id": 8,
            "question": f"Compare {titles[7]} and {titles[8]} in one sentence.",
            "expected_route": "rag",
            "expected_unknown": False,
        },
        {"id": 9, "question": f"What is {titles[9]}?", "expected_route": "rag", "expected_unknown": False},
        {
            "id": 10,
            "question": f"What is the phone number for {titles[0]}?",
            "expected_route": "rag",
            "expected_unknown": True,
        },
    ]


def _clip(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def judge_answer(
    question: str,
    answer: str,
    context: str,
    expected_unknown: bool,
) -> Dict[str, Any]:
    prompt = (
        "You are a strict evaluator for a retrieval-augmented QA system.\n"
        "Score the answer for relevance to the question and conciseness.\n"
        "If the context is empty, judge based on the question alone.\n"
        "Return JSON only with keys: relevance (1-5), conciseness (1-5), "
        "uses_context (yes/no/na), is_unknown (yes/no), notes.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Context (truncated):\n{_clip(context)}\n\n"
        f"Expected unknown: {expected_unknown}\n"
    )
    raw = chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=256)
    return _parse_judge_json(raw)


def _parse_judge_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return {
        "relevance": 0,
        "conciseness": 0,
        "uses_context": "na",
        "is_unknown": "no",
        "notes": "Failed to parse judge output",
        "raw": text,
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    avg_relevance = sum(float(r["relevance"]) for r in rows) / total
    avg_conciseness = sum(float(r["conciseness"]) for r in rows) / total
    pct_with_sources = sum(1 for r in rows if int(r["sources_count"]) > 0) / total
    pct_direct = sum(1 for r in rows if r["route"] == "direct") / total
    pct_unknown = sum(1 for r in rows if r["is_unknown"] == "yes") / total
    pct_expected_unknown_matched = (
        sum(1 for r in rows if r["expected_unknown"] and r["is_unknown"] == "yes")
        / max(1, sum(1 for r in rows if r["expected_unknown"]))
    )

    return {
        "total_questions": total,
        "avg_relevance": round(avg_relevance, 3),
        "avg_conciseness": round(avg_conciseness, 3),
        "pct_with_sources": round(pct_with_sources, 3),
        "pct_direct": round(pct_direct, 3),
        "pct_unknown": round(pct_unknown, 3),
        "pct_expected_unknown_matched": round(pct_expected_unknown_matched, 3),
    }


def main() -> None:
    load_dotenv()
    resources = initialize_pipeline()

    data_dir = Path(DATA_DIR)
    questions = build_questions(data_dir)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("data/evals")
    output_dir.mkdir(parents=True, exist_ok=True)
    questions_csv = output_dir / "questions_results.csv"
    experiments_csv = output_dir / "experiment_results.csv"

    rows: List[Dict[str, Any]] = []

    for item in questions:
        question = item["question"]
        info(f"Running eval question {item['id']}: {question}")
        result = run_query(question, resources)
        retrieval = result.get("retrieval", {}) or {}
        answer = result.get("answer", {}) or {}

        context = retrieval.get("context", "") or ""
        judge = judge_answer(
            question=question,
            answer=answer.get("answer", ""),
            context=context,
            expected_unknown=bool(item["expected_unknown"]),
        )

        rows.append(
            {
                "run_id": run_id,
                "trace_mode": "langsmith_api" if LANGGRAPH_USE_LANGSMITH_API else "local_only",
                "question_id": item["id"],
                "question": question,
                "expected_route": item["expected_route"],
                "expected_unknown": item["expected_unknown"],
                "route": result.get("route", ""),
                "route_reason": result.get("route_reason", ""),
                "retrieved_count": len(retrieval.get("retrieved", []) or []),
                "reranked_count": len(retrieval.get("reranked", []) or []),
                "sources_count": len(answer.get("sources", []) or []),
                "answer": answer.get("answer", ""),
                "relevance": judge.get("relevance", 0),
                "conciseness": judge.get("conciseness", 0),
                "uses_context": judge.get("uses_context", "na"),
                "is_unknown": judge.get("is_unknown", "no"),
                "judge_notes": judge.get("notes", ""),
            }
        )

    # Maintain a single questions_results.csv keyed by run_id.
    existing_rows: List[Dict[str, Any]] = []
    if questions_csv.exists():
        with questions_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = [row for row in reader if row.get("run_id") != run_id]

    fieldnames = list(rows[0].keys())
    with questions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(rows)

    summary = _aggregate(rows)
    summary_row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_mode": "langsmith_api" if LANGGRAPH_USE_LANGSMITH_API else "local_only",
        **summary,
    }

    write_header = not experiments_csv.exists()
    with experiments_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)

    info(f"Wrote question-level results to {questions_csv}")
    info(f"Wrote experiment summary to {experiments_csv}")


if __name__ == "__main__":
    main()
