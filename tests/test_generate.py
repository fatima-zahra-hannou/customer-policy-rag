"""Unit tests for pure functions in the generation module."""

from rag.generate import NOT_FOUND, abstain, format_passages


def test_abstain_builds_a_refusal():
    """abstain() returns a complete refusal with no sources."""
    result = abstain("une question", "low_retrieval_score", 0.02)

    assert result["answer"] == NOT_FOUND
    assert result["abstained"] is True
    assert result["reason"] == "low_retrieval_score"
    assert result["sources"] == []


def test_format_passages_numbers_and_labels_each_passage():
    """Each passage is numbered with its rank, page and section."""
    passages = [
        {"rank": 1, "page": 13, "section": "Article 6", "text": "Premier extrait."},
        {"rank": 2, "page": 14, "section": "Article 6", "text": "Second extrait."},
    ]

    formatted = format_passages(passages)

    assert "[1] (page 13, Article 6)" in formatted
    assert "[2] (page 14, Article 6)" in formatted
    assert "Premier extrait." in formatted
    assert "Second extrait." in formatted


def test_format_passages_preserves_order():
    """Passages appear in the order given, not resorted."""
    passages = [
        {"rank": 1, "page": 5, "section": "A", "text": "x"},
        {"rank": 2, "page": 6, "section": "B", "text": "y"},
    ]

    formatted = format_passages(passages)

    assert formatted.index("[1]") < formatted.index("[2]")
