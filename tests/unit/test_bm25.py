import pytest

from app.retrieval.sparse import build_bm25_index, bm25_scores


@pytest.mark.unit
def test_bm25_scores_basic():
    documents = [
        {"id": 1, "text": "apple banana"},
        {"id": 2, "text": "carrot tomato"},
    ]
    bm25, _ = build_bm25_index(documents)
    scores = bm25_scores(bm25, "apple", documents)

    assert set(scores.keys()) == {1, 2}
    assert scores[1] >= scores[2]
