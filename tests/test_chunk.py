"""Unit tests for the chunking logic."""

from rag.chunk import clean, find_source, skip_front_matter


def test_clean_collapses_blank_lines():
    """Three or more blank lines become exactly one."""
    text = "Article 1\n\n\n\nLe texte suit."
    assert clean(text) == "Article 1\n\nLe texte suit."


def test_clean_strips_trailing_spaces_before_newline():
    """Trailing spaces before a line break are removed."""
    text = "Une phrase.   \nLa suite."
    assert clean(text) == "Une phrase.\nLa suite."


def test_find_source_returns_correct_page():
    """An offset inside a page range resolves to that page's number."""
    page_starts = [(0, 1, "doc"), (100, 2, "doc"), (250, 3, "doc")]
    assert find_source(page_starts, 150) == (2, "doc")


def test_find_source_at_exact_boundary():
    """An offset exactly at a page's start belongs to that page."""
    page_starts = [(0, 1, "doc"), (100, 2, "doc")]
    assert find_source(page_starts, 100) == (2, "doc")


def test_skip_front_matter_removes_table_of_contents():
    """Everything before the body marker is dropped, and offsets shift."""
    text = "Article 1\nArticle 2\n\nPréambule\nLe vrai contenu commence ici."
    page_starts = [(0, 1, "doc"), (30, 2, "doc")]

    new_text, new_starts = skip_front_matter(text, page_starts)

    assert new_text.startswith("Préambule")
    assert new_starts[0][0] == 0


def test_skip_front_matter_keeps_text_when_marker_missing():
    """Without the body marker, the text is returned unchanged."""
    text = "Un document sans le mot clé attendu."
    page_starts = [(0, 1, "doc")]

    new_text, new_starts = skip_front_matter(text, page_starts)

    assert new_text == text
    assert new_starts == page_starts
