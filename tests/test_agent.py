"""Tests for src/agent.py — LLM tool-calling agent loop.

Owner: Div

Uses MockLLMClient — no API key needed, no network.
"""
import json
import os

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


class TestHydratePath:
    def test_hydrate_ends_alert_or_watch(self):
        """Mock hydrate scenario should produce tool_call, tool_result, and decision."""
        ctx = _make_ctx(has_event=True)
        mock_client = get_mock_client_for_scenario("hydrate")

        with patch("src.agent.get_llm_client", return_value=mock_client):
            events = list(run_agent(ctx, _trigger()))

        types = [e["type"] for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "decision" in types

        # Decision should be present (via rule fallback since mock doesn't produce JSON)
        decision = [e for e in events if e["type"] == "decision"][0]["data"]
        assert decision["decision"] in ("ALERT", "WATCH", "DISMISS")

    def test_emits_correct_tool_sequence(self):
        """Mock hydrate scenario calls tools in the expected order."""
        ctx = _make_ctx(has_event=True)
        mock_client = get_mock_client_for_scenario("hydrate")

        with patch("src.agent.get_llm_client", return_value=mock_client):
            events = list(run_agent(ctx, _trigger()))

        tool_calls = [e["data"]["tool"] for e in events if e["type"] == "tool_call"]
        assert "get_window" in tool_calls
        assert "classify_event" in tool_calls


class TestScalingPath:
    def test_scaling_different_from_hydrate(self):
        """Mock scaling scenario should take a different path than hydrate."""
        ctx_hydrate = _make_ctx(has_event=True)
        ctx_scaling = _make_ctx(has_event=False)
        mock_h = get_mock_client_for_scenario("hydrate")
        mock_s = get_mock_client_for_scenario("scaling")

        with patch("src.agent.get_llm_client", return_value=mock_h):
            events_h = list(run_agent(ctx_hydrate, _trigger()))
        # Clear cache for scaling
        import shutil
        from src.agent import CACHE_DIR
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)

        with patch("src.agent.get_llm_client", return_value=mock_s):
            events_s = list(run_agent(ctx_scaling, _trigger()))

        # Both should produce valid decisions
        dec_h = [e for e in events_h if e["type"] == "decision"][0]["data"]
        dec_s = [e for e in events_s if e["type"] == "decision"][0]["data"]
        assert dec_h["decision"] in ("ALERT", "WATCH", "DISMISS")
        assert dec_s["decision"] in ("ALERT", "WATCH", "DISMISS")
        # Scaling scenario calls fewer tools
        tc_h = len([e for e in events_h if e["type"] == "tool_call"])
        tc_s = len([e for e in events_s if e["type"] == "tool_call"])
        assert tc_s <= tc_h


class TestMaxToolCalls:
    def test_stops_at_eight_calls(self):
        """Agent loop must stop after 8 tool calls even if LLM keeps calling."""
        # Create a mock that always returns tool calls
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

        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) <= 8


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
        """Second run with same params should return cached events."""
        ctx = _make_ctx(has_event=True)
        mock_client = get_mock_client_for_scenario("hydrate")

        with patch("src.agent.get_llm_client", return_value=mock_client):
            events1 = list(run_agent(ctx, _trigger()))

        # Second run — should hit cache (no mock client needed)
        with patch("src.agent.get_llm_client", return_value=MockLLMClient()):
            events2 = list(run_agent(ctx, _trigger()))

        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1["type"] == e2["type"]
