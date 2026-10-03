"""Tests for Div's integration into Mico's fleet: LLM pool transport + RAG playbook.

Uses MockLLMClient — no API key needed, no network.
"""
import asyncio
import json
import os

os.environ["LLM_MODE"] = "mock"
os.environ.pop("OPENROUTER_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

import pytest
from unittest.mock import patch, MagicMock

from src.fleet_llm import (
    _tool, _fallback, _validate, _extract_playbook_refs,
    ASSESSMENT_SCHEMA, investigate,
)
from src.llm import make_fleet_transport, MockLLMClient, LLMResponse, ToolCall


class TestMakeFleetTransport:
    """4a: LLM pool transport adapter."""

    def test_mock_mode_returns_none(self):
        """In mock mode, transport=None so fleet_llm uses deterministic fallback."""
        with patch.dict(os.environ, {"LLM_MODE": "mock"}):
            transport = make_fleet_transport()
            assert transport is None

    def test_no_providers_returns_none(self):
        """When no API keys are set, transport=None."""
        env = {"LLM_MODE": "live", "GEMINI_API_KEY": "", "GROQ_API_KEY": "", "OPENROUTER_API_KEY": ""}
        with patch.dict(os.environ, env, clear=False):
            from src.llm import reset_provider_pool
            reset_provider_pool()
            transport = make_fleet_transport()
            assert transport is None
            reset_provider_pool()


class TestSearchPlaybookRAG:
    """4b: RAG playbook integration in fleet_llm._tool."""

    def test_search_playbook_returns_documents(self):
        """search_playbook should return docs with id, title, text fields."""
        snap = {"readings": [], "as_of": "2024-01-01T00:00:00"}
        result = _tool(snap, "search_playbook", {"query": "hydrate response"})
        assert result["available"] is True
        assert len(result["documents"]) > 0
        for doc in result["documents"]:
            assert "id" in doc
            assert "title" in doc
            assert "text" in doc

    def test_search_playbook_bm25_preferred(self):
        """With BM25 RAG available, source should be bm25_playbook."""
        snap = {"readings": [], "as_of": "2024-01-01T00:00:00"}
        result = _tool(snap, "search_playbook", {"query": "hydrate response checklist"})
        # If data/playbook/ has docs, source should be bm25_playbook
        # If not, falls back to project_playbook (inline)
        assert result["source"] in ("bm25_playbook", "project_playbook")

    def test_search_playbook_bad_query(self):
        """Empty or missing query should raise ValueError."""
        snap = {"readings": []}
        with pytest.raises(ValueError):
            _tool(snap, "search_playbook", {"query": ""})
        with pytest.raises(ValueError):
            _tool(snap, "search_playbook", {})


class TestPlaybookRefs:
    """4b: playbook_refs extraction and schema."""

    def test_playbook_refs_in_assessment_schema(self):
        """Assessment schema should accept optional playbook_refs."""
        assert "playbook_refs" in ASSESSMENT_SCHEMA["properties"]
        assert "playbook_refs" not in ASSESSMENT_SCHEMA["required"]

    def test_extract_from_evidence(self):
        """_extract_playbook_refs finds doc IDs from search_playbook results."""
        evidence = {
            "E1": {"id": "E1", "tool": "sensor_quality", "result": {"available": True}},
            "E2": {"id": "E2", "tool": "search_playbook", "result": {
                "available": True, "documents": [
                    {"id": "hydrate_response", "title": "Hydrate Response"},
                    {"id": "sensor_checks", "title": "Sensor Checks"},
                ],
            }},
        }
        refs = _extract_playbook_refs(evidence)
        assert refs == ["hydrate_response", "sensor_checks"]

    def test_extract_empty_when_no_playbook_tool(self):
        """No playbook tool calls -> empty refs."""
        evidence = {
            "E1": {"id": "E1", "tool": "sensor_quality", "result": {"available": True}},
        }
        assert _extract_playbook_refs(evidence) == []

    def test_fallback_includes_playbook_refs(self):
        """Fallback assessment should include playbook_refs."""
        snap = {
            "readings": [{"sensors": {"P-PDG": 280, "T-PDG": 85}, "t": "2024-01-01T00:00:00"}],
            "as_of": "2024-01-01T00:00:00",
            "model": {"available": False},
            "severity": "watch",
        }
        evidence = {
            "E1": {"id": "E1", "tool": "search_playbook", "result": {
                "available": True, "documents": [{"id": "test_doc"}],
            }},
        }
        result = _fallback(snap, "Test reason", evidence)
        assert "playbook_refs" in result
        assert result["playbook_refs"] == ["test_doc"]


class TestInvestigateDeterministic:
    """End-to-end: investigate in deterministic/fallback mode."""

    @pytest.mark.asyncio
    async def test_no_transport_uses_fallback(self):
        """With no transport (mock mode), investigate returns a deterministic result."""
        snap = {
            "well_id": "WELL-00001",
            "as_of": "2024-01-01T00:00:00",
            "severity": "watch",
            "readings": [{"t": "2024-01-01T00:00:00", "sensors": {"P-PDG": 280, "T-PDG": 85, "P-TPT": 278}}],
            "model": {"available": True, "version": "test", "scores": {"hydrate": 0.1, "lookalike": 0.05, "normal": 0.85}},
        }
        events = []
        result = await investigate(snap, lambda e: events.append(e), transport=None)
        assert result["status"] in ("watch", "normal", "attention", "telemetry")
        assert "playbook_refs" in result
        assert result["well_id"] == "WELL-00001"
