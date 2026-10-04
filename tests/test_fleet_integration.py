"""Div's integration into the fleet: provider-pool transport, live investigations and the RAG playbook.

Provider calls are mocked at ProviderPool.chat — no API key value is used and no network is touched.
"""
import asyncio
import json

import pytest

from src.fleet_llm import (
    _tool, _fallback, _extract_playbook_refs,
    ASSESSMENT_SCHEMA, investigate,
)
from src import llm
from src.llm import make_fleet_transport, LLMResponse, ToolCall


def pool_reply(*calls, model="llama-3.3-70b-versatile"):
    return LLMResponse(content=None, tool_calls=[ToolCall(id=f"call-{i}", name=name, arguments=args) for i, (name, args) in enumerate(calls)],
                       usage={"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}, model=model, provider="groq")


class TestMakeFleetTransport:
    """LLM pool transport adapter."""

    def test_mock_mode_returns_none(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-only")
        monkeypatch.setenv("LLM_MODE", "mock")
        assert make_fleet_transport() is None
        assert llm.configured_providers() == []

    def test_no_providers_returns_none(self):
        assert make_fleet_transport() is None

    def test_transport_converts_pool_response_to_openai_envelope(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-only")
        seen = []
        def chat(self, **kwargs):
            seen.append(kwargs)
            return pool_reply(("search_playbook", {"query": "hydrate"}))
        monkeypatch.setattr(llm.ProviderPool, "chat", chat)
        transport = make_fleet_transport()
        forced = {"type": "function", "function": {"name": "submit_assessment"}}
        reply = asyncio.run(transport({"messages": [{"role": "user", "content": "x"}], "tools": [], "tool_choice": forced}, 5))
        assert seen[0]["tool_choice"] == forced
        assert reply["model"] == "llama-3.3-70b-versatile" and reply["provider"] == "groq"
        call = reply["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "search_playbook"
        assert json.loads(call["function"]["arguments"]) == {"query": "hydrate"}
        assert reply["usage"]["cost"] == 0

    def test_exhausted_pool_raises_provider_failure(self, monkeypatch):
        from src.fleet_llm import ProviderFailure
        monkeypatch.setenv("GEMINI_API_KEY", "test-only")
        monkeypatch.setattr(llm.ProviderPool, "chat", lambda self, **kw: LLMResponse(None, [], {}, model="exhausted", provider="all"))
        with pytest.raises(ProviderFailure):
            asyncio.run(make_fleet_transport()({"messages": []}, 5))


def test_use_llm_session_runs_live_investigation_through_provider_pool(monkeypatch):
    """use_llm=True + configured provider: the fleet uses the pool transport, BM25 playbook and validation."""
    from backend import fleet
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    assessment = {"status": "attention", "diagnosis": "hydrate_suspected", "brief": "Model risk and pressure evidence support a hydrate review.",
                  "action": "review_hydrate", "recheck_minutes": 5, "evidence_ids": ["E1", "E2", "E3", "E4"],
                  "alternative": "A restriction remains possible.", "missing_evidence": []}
    planned = [pool_reply(("search_playbook", {"query": "hydrate review"})), pool_reply(("submit_assessment", assessment))]
    monkeypatch.setattr(llm.ProviderPool, "chat", lambda self, **kw: planned.pop(0))
    fleet.SESSIONS.clear(); fleet.ATTEMPTS.clear(); fleet.PROVIDER_CIRCUIT.update(until=0.0, reason="")

    async def scenario():
        session = fleet.FleetSession(use_llm=True)
        fleet.SESSIONS[session.id] = session
        assert session.snapshot()["use_llm"] is True and session.snapshot()["llm_providers"] == ["groq"]
        await session.prepare()
        for _ in range(40):
            session.advance()
        await session.investigate_well("WELL-00019")
        return session

    session = asyncio.run(scenario())
    well = session.wells["WELL-00019"]
    assert well["status"] == "attention"
    assert well["investigation"] == "complete"
    assert well["assessment"]["source"] == "live"
    assert well["assessment"]["summary"].startswith("Model risk")
    assert well["assessment"]["playbook_refs"]
    assert well["assessment"]["checks"]
    assert session.snapshot()["agent_mode"] == "live"
    assert session.requests_used == 2
    result = next(e["result"] for e in reversed(session.audit) if e["kind"] == "assessment" and "result" in e)
    assert result["metadata"]["model"] == "llama-3.3-70b-versatile" and result["metadata"]["provider"] == "groq"
    fleet.SESSIONS.clear(); fleet.ATTEMPTS.clear()


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
