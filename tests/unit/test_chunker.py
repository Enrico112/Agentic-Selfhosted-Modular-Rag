import pytest

from app.ingestion.markdown_chunker import chunk_markdown


@pytest.mark.unit
def test_chunk_markdown_basic(tmp_path):
    md = "# Title\n\nIntro paragraph.\n\n## Section A\n\nSentence one. Sentence two.\n"
    path = tmp_path / "doc.md"
    path.write_text(md, encoding="utf-8")

    chunks = chunk_markdown(path, max_tokens=50, overlap_ratio=0.1, debug=False)
    assert chunks
    first = chunks[0]
    assert "text" in first
    assert "metadata" in first
    assert first["metadata"].get("file_path") == str(path)
    assert first["metadata"].get("section")
