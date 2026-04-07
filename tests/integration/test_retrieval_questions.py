from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.llm.client import chat
from app.rag.pipeline import initialize_pipeline, run_query


DATA_DIR = Path("data/goodwiki_markdown_sample")

QUESTIONS: List[Dict[str, str]] = [
    {
        "id": "4f_case",
        "query": "What happened in the 4F case on February 4, 2006, and how many people were arrested?",
        "expected": "A policeman outside a rave in Barcelona was paralyzed after being hit by a falling object, and nine people were arrested.",
    },
    {
        "id": "subway_construction",
        "query": "In what year did development of New York City's first subway line begin, and which act enabled it?",
        "expected": "Development began in 1894, enabled by the Rapid Transit Act passed by the New York State Legislature.",
    },
    {
        "id": "pub_design",
        "query": "Which earlier building influenced the 17-storey PUB Building's design, and who led its winning design?",
        "expected": "It was influenced by Boston City Hall (Gerhard M. Kallmann), with Le Corbusier's Sainte Marie de La Tourette also cited as an influence, and the winning design was by Group 2 Architects.",
    },
    {
        "id": "178th_street",
        "query": "Why did William Jerome Daly oppose a station at 178th Street, and what operational issue did he cite?",
        "expected": "He said a station there would prevent express service from operating past 71st Avenue, limiting express operations; he noted express service could run to Parsons Boulevard from a 169th Street terminal and extend further if the line went to Springfield Boulevard.",
    },
    {
        "id": "1812_met_history",
        "query": "Why was the hurricane's track uncertain in August 1812, and what caused the lack of observations?",
        "expected": "The War of 1812 led to a British blockade of American ships, resulting in few observations and an uncertain track.",
    },
    {
        "id": "1890_season_summary",
        "query": "In the 1890 Atlantic hurricane season, how many tropical cyclones reached hurricane status, and what was the strongest storm's peak wind speed?",
        "expected": "Two reached hurricane status, and the strongest peaked at Category 3 with 120 mph (195 km/h) winds.",
    },
]


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
        "is_correct": "no",
        "missing": "Failed to parse judge output",
        "notes": text,
    }


def _judge_answer(question: str, expected: str, answer: str) -> Dict[str, Any]:
    prompt = (
        "You are a strict evaluator for a retrieval QA system.\n"
        "Decide whether the answer matches the expected facts.\n"
        "Extra context is allowed and should NOT be penalized unless it contradicts the expected facts.\n"
        "If the answer includes the expected facts (even with additional detail), mark is_correct=yes.\n"
        "Return JSON only with keys: is_correct (yes/no), missing, notes.\n\n"
        f"Question:\n{question}\n\n"
        f"Expected answer:\n{expected}\n\n"
        f"Model answer:\n{answer}\n"
    )
    raw = chat([{"role": "user", "content": prompt}], temperature=0.0, num_predict=256)
    return _parse_judge_json(raw)


@pytest.fixture(scope="session")
def resources():
    if not DATA_DIR.exists():
        pytest.skip(f"Missing data directory: {DATA_DIR}")
    try:
        return initialize_pipeline()
    except Exception as exc:
        pytest.skip(f"Pipeline init failed: {exc}")


@pytest.mark.integration
@pytest.mark.parametrize("case", QUESTIONS)
def test_retrieval_questions_with_llm_judge(case, resources):
    result = run_query(case["query"], resources)
    answer = result.get("answer", {}).get("answer", "")
    assert answer, "Expected non-empty answer"

    judge = _judge_answer(case["query"], case["expected"], answer)
    assert judge.get("is_correct") == "yes", f"Judge failed: {judge}"
