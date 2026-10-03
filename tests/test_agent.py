"""Tests for src/agent.py — two-round LLM tool-calling agent loop.

Owner: Div

Uses MockLLMClient — no API key needed, no network.
"""
import json
import os
import shutil

os.environ["LLM_MODE"] = "mock"
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["OPENROUTER_MODEL"] = ""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.agent import (
    run_agent, cache_decision, get_cached_decision,
    _build_rule_fallback, get_mock_client_for_scenario,
    CACHE_DIR, LLM_MAX_REQUESTS_PER_RUN,
)
from src.llm import MockLLMClient, LLMResponse, ToolCall
from src.tools import AgentContext


def _make_ctx(n_minutes=100, minute_index=None, has_event=True):
    if minute_index is None:
        minute_index = n_minutes - 1
    idx = pd.date_range("2024-01-01", periods=n_minutes, freq="1min")
    df = pd.DataFrame({
        "P-PDG": np.linspace(280, 260 if has_event else 280, n_minutes),
        "T-PDG": np.linspace(85, 80 if has_event else 85, n_minutes),
        "P-TPT": np.linspace(278, 258 if has_event else 278, n_minutes),
        "T-TPT": np.linspace(84, 79 if has_event else 84, n_minutes),
        "P-MON-CKP": np.linspace(276, 256 if has_event else 276, n_minutes),
        "P-JUS-CKP": np.full(n_minutes, np.nan),
        "T-JUS-CKP": np.full(n_minutes, np.nan),
        "ABER-CKP": np.full(n_minutes, 30.0),
        "QGL": np.linspace(12, 10 if has_event else 12, n_minutes),
        "phase": ["normal"] * n_minutes,
        "margin_C": np.linspace(5, -2 if has_event else 5, n_minutes),
    }, index=idx)
    return AgentContext(
        instance_id="test_instance",
        well_id="WELL-TEST",
        minute_index=minute_index,
        df=df,
    )


def _trigger(t="2024-01-01T01:39:00"):
    return {"t": t, "reason": "p_hydrate >= 0.5 for 3+ min", "score": 0.72}


def _cleanup_cache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)


class TestTwoRoundFlow:
    def test_hydrate_ends_alert(self):
        """Mock hydrate scenario should produce ALERT with search_playbook."""
        _cleanup_cache()
        ctx = _make_ctx(has_event=True)
        mock_client = get_mock_client_for_scenario("hydrate")

        with patch("src.agent.get_llm_client", return_value=mock_client):
            events = list(run_agent(ctx, _trigger()))

        types = [e["type"] for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "decision" in types

        decision = [e for e in events if e["type"] == "decision"][0]["data"]
        assert decision["decision"] == "ALERT"

    def test_scaling_ends_dismiss(self):
        """Mock scaling scenario should produce DISMISS."""
        _cleanup_cache()
        ctx = _make_ctx(has_event=False)
        mock_client = get_mock_client_for_scenario("scaling")

        with patch("src.agent.get_llm_client", return_value=mock_client):
            events = list(run_agent(ctx, _trigger()))

        decision = [e for e in events if e["type"] == "decision"][0]["data"]
        assert decision["decision"] == "DISMISS"

    def test_pre_run_tools_labeled_system(self):
        """Pre-run tools (get_window, classify_event) should have requested_by=system."""
        _cleanup_cache()
        ctx = _make_ctx(has_event=True)
        mock_client = get_mock_client_for_scenario("hydrate")

        with patch("src.agent.get_llm_client", return_value=mock_client):
            events = list(run_agent(ctx, _trigger()))

        tool_calls = [e for e in events if e["type"] == "tool_call"]
        system_calls = [tc for tc in tool_calls if tc["data"].get("requested_by") == "system"]
        agent_calls = [tc for tc in tool_calls if tc["data"].get("requested_by") == "agent"]

        system_tools = [tc["data"]["tool"] for tc in system_calls]
        assert "get_window" in system_tools
        assert "classify_event" in system_tools
        assert len(agent_calls) > 0

    def test_uses_at_most_max_requests(self):
        """Agent should use at most LLM_MAX_REQUESTS_PER_RUN LLM requests."""
        _cleanup_cache()
        infinite_calls = [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id=f"c{i}", name="get_window", arguments={"minutes": 60})],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model="mock",
            )
            for i in range(20)
        ]
        mock_client = MockLLMClient(planned=infinite_calls)

        ctx = _make_ctx()
        with patch("src.agent.get_llm_client", return_value=mock_client):
            events = list(run_agent(ctx, _trigger()))

        assert len(mock_client.calls) <= LLM_MAX_REQUESTS_PER_RUN

    def test_search_playbook_in_hydrate_trace(self):
        """Mock hydrate scenario should call search_playbook."""
        _cleanup_cache()
        ctx = _make_ctx(has_event=True)
        mock_client = get_mock_client_for_scenario("hydrate")

        with patch("src.agent.get_llm_client", return_value=mock_client):
            events = list(run_agent(ctx, _trigger()))

        tool_calls = [e["data"]["tool"] for e in events if e["type"] == "tool_call"]
        assert "search_playbook" in tool_calls


class TestRuleFallback:
    def test_fallback_produces_valid_decision(self):
        ctx = _make_ctx(has_event=True)
        tool_results = {
            "classify_event": {"available": True, "source": "heuristic_fallback",
                               "p_hydrate": 0.65, "p_lookalike": 0.15, "p_normal": 0.20},
        }
        decision = _build_rule_fallback(ctx, tool_results)
        assert decision["decision"] in ("ALERT", "WATCH", "DISMISS")
        assert decision["source"] == "rule_fallback"
        assert len(decision["evidence"]) > 0

    def test_fallback_dismiss_for_normal(self):
        ctx = _make_ctx(has_event=False)
        tool_results = {
            "classify_event": {"available": True, "source": "heuristic_fallback",
                               "p_hydrate": 0.05, "p_lookalike": 0.10, "p_normal": 0.85},
        }
        decision = _build_rule_fallback(ctx, tool_results)
        assert decision["decision"] == "DISMISS"


class TestCaching:
    def test_cache_hit(self):
        decision = {"decision": "DISMISS", "rationale": "test"}
        cache_decision("WELL-TEST-CACHE", "2024-01-07T08:00", decision)
        cached = get_cached_decision("WELL-TEST-CACHE", "2024-01-07T08:00")
        assert cached is not None
        assert cached["decision"] == "DISMISS"

    def test_cache_miss(self):
        cached = get_cached_decision("NONEXISTENT", "2024-01-07T00:00")
        assert cached is None

    def test_agent_cache_returns_identical_events(self):
        _cleanup_cache()
        ctx = _make_ctx(has_event=True)
        mock_client = get_mock_client_for_scenario("hydrate")

        with patch("src.agent.get_llm_client", return_value=mock_client):
            events1 = list(run_agent(ctx, _trigger()))

        with patch("src.agent.get_llm_client", return_value=MockLLMClient()):
            events2 = list(run_agent(ctx, _trigger()))

        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1["type"] == e2["type"]

    def test_cache_hit_uses_zero_requests(self):
        _cleanup_cache()
        ctx = _make_ctx(has_event=True)
        mock_client = get_mock_client_for_scenario("hydrate")

        with patch("src.agent.get_llm_client", return_value=mock_client):
            list(run_agent(ctx, _trigger()))

        fresh_mock = MockLLMClient()
        with patch("src.agent.get_llm_client", return_value=fresh_mock):
            events2 = list(run_agent(ctx, _trigger()))

        assert len(fresh_mock.calls) == 0
        assert len(events2) > 0


class TestCacheOnly:
    def test_cache_only_never_calls_llm(self):
        _cleanup_cache()
        ctx = _make_ctx(has_event=False)
        mock_client = MockLLMClient()

        with patch("src.agent.get_llm_client", return_value=mock_client):
            events = list(run_agent(ctx, _trigger(), cache_only=True))

        assert len(mock_client.calls) == 0
        decisions = [e for e in events if e["type"] == "decision"]
        assert len(decisions) == 1
        assert decisions[0]["data"]["source"] == "rule_fallback"

    def test_cache_only_uses_cache_when_available(self):
        _cleanup_cache()
        ctx = _make_ctx(has_event=True)
        mock_client = get_mock_client_for_scenario("hydrate")

        with patch("src.agent.get_llm_client", return_value=mock_client):
            events1 = list(run_agent(ctx, _trigger()))

        fresh_mock = MockLLMClient()
        with patch("src.agent.get_llm_client", return_value=fresh_mock):
            events2 = list(run_agent(ctx, _trigger(), cache_only=True))

        assert len(fresh_mock.calls) == 0
        assert len(events1) == len(events2)
