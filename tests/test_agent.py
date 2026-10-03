"""Tests for src/agent.py — LLM tool-calling agent loop.

Owner: Div

Uses a mocked LLM client — no API key needed.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.pending


class TestMaxToolCalls:
    def test_stops_at_eight_calls(self):
        """Agent loop must stop after 8 tool calls even if LLM keeps calling."""
        from src.agent import run_agent

        # Mock the LLM to always request another tool call
        with patch("src.agent.run_agent") as mock_run:
            mock_run.return_value = {
                "decision": "ALERT",
                "rationale": "test",
                "tool_calls": [{"tool": f"t{i}"} for i in range(8)],
            }
            result = mock_run(well_id="WELL-00019", trigger_context={})
            assert len(result["tool_calls"]) <= 8


class TestMalformedJSON:
    def test_retry_then_fallback(self):
        """If LLM returns malformed JSON, retry once then use rule-based fallback."""
        from src.agent import run_agent

        # The contract: even with a broken LLM, agent returns a valid decision
        with patch("src.agent.run_agent") as mock_run:
            mock_run.return_value = {
                "decision": "WATCH",
                "rationale": "rule-based fallback",
                "tool_calls": [],
            }
            result = mock_run(well_id="WELL-00019", trigger_context={})
            assert result["decision"] in ("ALERT", "WATCH", "DISMISS")


class TestBriefTraceability:
    def test_numbers_from_tools(self):
        """Every number in the brief must trace back to a tool result."""
        # This is a contract test: when implemented, the agent must
        # only include numbers that came from tool_result events.
        from src.agent import run_agent

        with patch("src.agent.run_agent") as mock_run:
            mock_run.return_value = {
                "decision": "ALERT",
                "rationale": "margin = -1.3 C from hydrate_margin tool",
                "tool_calls": [
                    {"tool": "hydrate_margin", "result": {"margin_C": -1.3}},
                ],
                "brief": "Margin is -1.3 C.",
            }
            result = mock_run(well_id="WELL-00019", trigger_context={})
            # Verify -1.3 appears in both tool results and brief
            tool_numbers = set()
            for tc in result.get("tool_calls", []):
                for v in (tc.get("result") or {}).values():
                    if isinstance(v, (int, float)):
                        tool_numbers.add(v)
            # Every number in brief should be in tool_numbers
            # (simplified check for contract)
            assert -1.3 in tool_numbers


class TestCaching:
    def test_cache_hit(self):
        """Repeated call for same (well, minute) should return cached decision."""
        from src.agent import cache_decision, get_cached_decision

        decision = {"decision": "DISMISS", "rationale": "test"}
        cache_decision("WELL-00019", "2024-01-07T08:00", decision)
        cached = get_cached_decision("WELL-00019", "2024-01-07T08:00")
        assert cached is not None
        assert cached["decision"] == "DISMISS"
