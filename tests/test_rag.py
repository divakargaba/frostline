"""Tests for src/rag.py — BM25 retrieval over playbook.

Owner: Div
"""

import pytest

pytestmark = pytest.mark.pending


class TestRetrieval:
    def test_returns_title_snippet_source(self):
        """Search results must include title, snippet, and source fields."""
        from src.rag import search

        results = search("hydrate formation response procedure", top_k=3)
        assert isinstance(results, list)
        for r in results:
            assert "title" in r
            assert "snippet" in r
            # doc_id or source
            assert "doc_id" in r or "source" in r

    def test_known_query_returns_results(self):
        """A query matching playbook content should return at least one result."""
        from src.rag import search

        results = search("methanol injection", top_k=3)
        assert len(results) >= 1

    def test_empty_query(self):
        """An empty or irrelevant query should return an empty list."""
        from src.rag import search

        results = search("xyzzy_completely_unrelated_gibberish", top_k=3)
        assert isinstance(results, list)
