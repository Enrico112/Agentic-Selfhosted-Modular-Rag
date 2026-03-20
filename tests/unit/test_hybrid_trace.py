import pytest

from app.retrieval.hybrid import _top_k_scores


@pytest.mark.unit
def test_top_k_scores_includes_file_path():
    scores = {1: 0.9, 2: 0.1}
    payloads = {
        1: {"metadata": {"file_path": "file_a.md"}},
        2: {"metadata": {"file_path": "file_b.md"}},
    }

    top = _top_k_scores(scores, payloads, k=2)
    assert top[0]["file_path"] == "file_a.md"
    assert top[1]["file_path"] == "file_b.md"
