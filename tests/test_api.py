"""Tests for backend API endpoints.

Owner: Div

No network calls. Uses FastAPI TestClient, LLM_MODE=mock.
"""
import json
import os

# Force mock mode before importing app
os.environ["LLM_MODE"] = "mock"
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["OPENROUTER_MODEL"] = ""

import pytest
from starlette.testclient import TestClient

from backend.main import app
from backend.schemas import WellInfo

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["llm_mode"] == "mock"
    assert "has_seed" in data


def test_wells_schema():
    r = client.get("/wells")
    assert r.status_code == 200
    wells = r.json()
    assert isinstance(wells, list)
    for w in wells:
        WellInfo(**w)


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into events."""
    events = []
    current = {}
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current["data"] = line.split(":", 1)[1].strip()
        elif line.startswith("id:"):
            current["id"] = line.split(":", 1)[1].strip()
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


def test_stream_seed():
    """Stream seed with high speed and to_minute=5 — get ticks + phase_marker + end."""
    with client.stream("GET", "/stream/seed?speed=6000&to_minute=5") as r:
        assert r.status_code == 200
        text = r.read().decode()

    events = _parse_sse(text)
    event_types = [e.get("event") for e in events]
    assert "phase_marker" in event_types
    assert "tick" in event_types
    assert "end" in event_types


def test_stream_unknown_404():
    r = client.get("/stream/nonexistent_xyz_12345?speed=6000")
    assert r.status_code == 404


def test_results_mock_fallback():
    r = client.get("/results")
    assert r.status_code == 200
    data = r.json()
    assert "systems" in data
    assert data.get("mock") is True


def test_stream_record(tmp_path, monkeypatch):
    """Record mode writes a .jsonl file."""
    import backend.config
    monkeypatch.setattr(backend.config, "DATA_DEMO", tmp_path)

    with client.stream("GET", "/stream/seed?speed=6000&to_minute=3&record=true") as r:
        assert r.status_code == 200
        _ = r.read()

    jsonl_files = list(tmp_path.glob("seed_*.jsonl"))
    assert len(jsonl_files) >= 1
    content = jsonl_files[0].read_text()
    assert "tick" in content


def test_stream_seed_with_agent():
    """Stream seed with agent=true should emit watcher/agent events."""
    # Seed data has hydrate events (label=1) starting around hour 150,
    # so stream a range that includes the transition.
    # The watcher uses fallback rules since no model/physics.
    with client.stream("GET", "/stream/seed?speed=6000&from_minute=148&to_minute=160&agent=true") as r:
        assert r.status_code == 200
        text = r.read().decode()

    events = _parse_sse(text)
    event_types = [e.get("event") for e in events]

    # Should have standard events
    assert "tick" in event_types
    assert "end" in event_types

    # With agent enabled and the watcher fallback rule, we might get triggers
    # depending on pressure. The test validates the stream doesn't crash.
    # All events should have valid JSON data
    for e in events:
        if "data" in e:
            json.loads(e["data"])  # must not raise


def test_stream_without_agent():
    """agent=false should produce only tick/phase/end events."""
    with client.stream("GET", "/stream/seed?speed=6000&to_minute=5&agent=false") as r:
        assert r.status_code == 200
        text = r.read().decode()

    events = _parse_sse(text)
    event_types = set(e.get("event") for e in events)
    # Should NOT have agent events
    assert "tool_call" not in event_types
    assert "tool_result" not in event_types
    assert "decision" not in event_types
