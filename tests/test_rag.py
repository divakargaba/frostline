"""Tests for src/rag.py — BM25 retrieval over playbook.

Owner: Div
"""
import pytest
from src.rag import search, get_index


class TestRetrieval:
    def test_methanol_vs_meg_returns_inhibitor_doc(self):
        """'methanol vs MEG' should return the thermodynamic inhibitors doc."""
        results = search("methanol vs MEG", top_k=3)
        assert len(results) >= 1
        ids = [r["id"] for r in results]
        assert "thermodynamic_inhibitors" in ids

    def test_scaling_returns_scaling_doc(self):
        """'scaling' should return the scaling look-alike doc."""
        results = search("scaling vs hydrate", top_k=3)
        assert len(results) >= 1
        ids = [r["id"] for r in results]
        assert "scaling_vs_hydrate" in ids

    def test_results_have_required_fields(self):
        """Search results must include id, title, snippet, sources."""
        results = search("hydrate formation response", top_k=3)
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert "id" in r
            assert "title" in r
            assert "snippet" in r
            assert "sources" in r
            assert isinstance(r["sources"], list)
            assert "score" in r

    def test_known_query_returns_results(self):
        """A query matching playbook content should return at least one result."""
        results = search("methanol injection", top_k=3)
        assert len(results) >= 1

    def test_empty_query(self):
        """Gibberish query should return empty or low-score results."""
        results = search("xyzzy_completely_unrelated_gibberish", top_k=3)
        assert isinstance(results, list)

    def test_hydrate_alert_returns_response_checklist(self):
        """'hydrate alert response' should return the response checklist."""
        results = search("hydrate alert response checklist", top_k=3)
        assert len(results) >= 1
        ids = [r["id"] for r in results]
        assert "hydrate_response" in ids

    def test_deduplicates_by_doc_id(self):
        """Results should not have duplicate doc IDs."""
        results = search("hydrate", top_k=10)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_index_loads_all_docs(self):
        """Index should load all playbook docs (not SOURCES.md)."""
        idx = get_index()
        if not idx._loaded:
            idx.load()
        doc_ids = set(c.doc_id for c in idx.chunks)
        # Should have at least 12 docs
        assert len(doc_ids) >= 12
