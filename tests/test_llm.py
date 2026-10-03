"""Tests for src/llm.py — MockLLMClient, model rotation, budget tracking.

No network calls. All tests use MockLLMClient.
"""
import os
import pytest
from src.llm import MockLLMClient, LLMResponse, ToolCall, get_llm_client, LLMUsageTracker


class TestMockLLMClient:
    def test_returns_planned_responses(self):
        planned = [
            LLMResponse(content="first", tool_calls=[], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
            LLMResponse(content="second", tool_calls=[], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        ]
        client = MockLLMClient(planned=planned)
        r1 = client.chat([{"role": "user", "content": "hi"}])
        assert r1.content == "first"
        r2 = client.chat([{"role": "user", "content": "hello"}])
        assert r2.content == "second"

    def test_fallback_when_no_planned(self):
        client = MockLLMClient()
        r = client.chat([{"role": "user", "content": "hi"}])
        assert "Mock" in r.content
        assert r.tool_calls == []

    def test_records_calls(self):
        client = MockLLMClient()
        client.chat([{"role": "user", "content": "test"}], tools=[{"type": "function"}])
        assert len(client.calls) == 1
        assert client.calls[0]["tools"] == [{"type": "function"}]

    def test_tool_call_response(self):
        tc = ToolCall(id="c1", name="get_time", arguments={})
        planned = [LLMResponse(content=None, tool_calls=[tc], usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})]
        client = MockLLMClient(planned=planned)
        r = client.chat([{"role": "user", "content": "time?"}])
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "get_time"


class TestGetLLMClient:
    def test_falls_back_to_mock_without_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        monkeypatch.setenv("OPENROUTER_MODEL", "")
        monkeypatch.setenv("LLM_MODE", "")
        import importlib
        import backend.config
        importlib.reload(backend.config)
        client = get_llm_client()
        assert isinstance(client, MockLLMClient)

    def test_mock_mode_forced(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setenv("OPENROUTER_MODEL", "test/model")
        monkeypatch.setenv("LLM_MODE", "mock")
        import importlib
        import backend.config
        importlib.reload(backend.config)
        client = get_llm_client()
        assert isinstance(client, MockLLMClient)


class TestLLMResponse:
    def test_dataclass_fields(self):
        r = LLMResponse(
            content="hello",
            tool_calls=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model="test",
            cost_estimate=0.001,
        )
        assert r.content == "hello"
        assert r.model == "test"
        assert r.cost_estimate == 0.001


class TestUsageTracker:
    def test_records_and_counts(self, tmp_path, monkeypatch):
        import src.llm
        monkeypatch.setattr(src.llm, "USAGE_PATH", tmp_path / "usage.json")
        tracker = LLMUsageTracker()
        tracker.record("model-a", 100)
        tracker.record("model-b", 200)
        assert tracker.requests_last_minute() == 2
        assert tracker.requests_last_day() == 2
        summary = tracker.summary()
        assert summary["total_requests"] == 2
        assert summary["total_tokens"] == 300

    def test_summary_empty(self, tmp_path, monkeypatch):
        import src.llm
        monkeypatch.setattr(src.llm, "USAGE_PATH", tmp_path / "usage.json")
        tracker = LLMUsageTracker()
        s = tracker.summary()
        assert s["requests_last_minute"] == 0
        assert s["total_tokens"] == 0
