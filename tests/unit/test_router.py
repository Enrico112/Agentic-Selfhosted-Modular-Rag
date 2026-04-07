import pytest

from app.agents.router_agent import route_query


@pytest.mark.unit
def test_route_query_summary():
    result = route_query("summarize the policy")
    assert result["route"] == "summarize"


@pytest.mark.unit
def test_route_query_compare():
    result = route_query("compare apples vs oranges")
    assert result["route"] == "rag"


@pytest.mark.unit
def test_route_query_question_words():
    result = route_query("who wrote the report")
    assert result["route"] == "rag"


@pytest.mark.unit
def test_route_query_default_direct():
    result = route_query("hello there")
    assert result["route"] == "direct"
