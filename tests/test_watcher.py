"""Tests for src/watcher.py — per-tick anomaly trigger.

Owner: Div
"""
import pandas as pd
import pytest

from src.watcher import Watcher, should_trigger, record_decision, get_next_recheck


class TestTrigger:
    def test_triggers_above_threshold(self):
        """Watcher should trigger when p_hydrate exceeds threshold for 3+ min."""
        w = Watcher()
        tick = {"t": "2024-01-07T08:00", "P-TPT": 250.0, "T-TPT": 5.0,
                "p_hydrate": 0.85, "margin_C": None, "sensors": {}}
        # Need 3 consecutive high ticks
        w.check_tick(tick, "WELL-00019", 0)
        w.check_tick(tick, "WELL-00019", 1)
        result = w.check_tick(tick, "WELL-00019", 2)
        assert result is not None
        assert result["score"] > 0

    def test_no_trigger_normal(self):
        """Watcher should not trigger during clearly normal conditions."""
        w = Watcher()
        tick = {"t": "2024-01-07T08:00", "P-TPT": 280.0, "T-TPT": 85.0,
                "p_hydrate": 0.05, "margin_C": 5.0, "sensors": {}}
        result = w.check_tick(tick, "WELL-00019", 0)
        assert result is None

    def test_margin_danger_triggers(self):
        """Negative margin should trigger."""
        w = Watcher()
        tick = {"t": "2024-01-07T08:00", "p_hydrate": None,
                "margin_C": -1.0, "sensors": {}}
        result = w.check_tick(tick, "WELL-00019", 0)
        assert result is not None

    def test_fallback_rule_no_model_no_physics(self):
        """When both model and physics are missing, pressure-based fallback fires."""
        w = Watcher()
        tick = {"t": "2024-01-07T08:00", "p_hydrate": None, "margin_C": None,
                "sensors": {"P_TPT_bar": 250.0}}
        result = w.check_tick(tick, "WELL-00019", 0)
        assert result is not None
        assert "Pressure fallback" in result["reason"]


class TestCooldown:
    def test_cooldown_prevents_retrigger(self):
        w = Watcher()
        tick = {"t": "2024-01-07T08:00", "p_hydrate": 0.85, "margin_C": None, "sensors": {}}
        # Trigger first
        w.check_tick(tick, "WELL-00019", 0)
        w.check_tick(tick, "WELL-00019", 1)
        result = w.check_tick(tick, "WELL-00019", 2)
        assert result is not None

        # Record decision
        w.record_decision("WELL-00019", "ALERT", 2, score=0.85)

        # 2 minutes later — still in cooldown (default 30 min)
        result2 = w.check_tick(tick, "WELL-00019", 4)
        assert result2 is None

    def test_cooldown_expires(self):
        w = Watcher()
        tick = {"t": "2024-01-07T08:00", "p_hydrate": 0.85, "margin_C": None, "sensors": {}}
        w.record_decision("WELL-00019", "ALERT", 0, score=0.85)

        # After cooldown (31 min later)
        w.check_tick(tick, "WELL-00019", 31)
        w.check_tick(tick, "WELL-00019", 32)
        result = w.check_tick(tick, "WELL-00019", 33)
        assert result is not None


class TestRecheck:
    def test_recheck_after_watch(self):
        w = Watcher()
        w.record_decision("WELL-00019", "WATCH", 0, score=0.5, recheck_min=15)

        tick = {"t": "2024-01-07T08:15", "p_hydrate": 0.3, "margin_C": None, "sensors": {}}
        # At minute 15 (recheck due), should trigger even with low score
        result = w.check_tick(tick, "WELL-00019", 15)
        assert result is not None

    def test_no_recheck_after_alert(self):
        w = Watcher()
        w.record_decision("WELL-00019", "ALERT", 0, score=0.9)
        st = w._state("WELL-00019")
        assert st.recheck_at_minute is None


class TestThresholds:
    def test_default_thresholds(self):
        from src.tools import DEFAULT_THRESHOLDS
        w = Watcher()
        assert w.thresholds["watch_threshold"] == DEFAULT_THRESHOLDS["watch_threshold"]
        assert w.thresholds["cooldown_min"] == DEFAULT_THRESHOLDS["cooldown_min"]


# Legacy API tests (existing test signatures)
class TestLegacyAPI:
    def test_should_trigger_true(self):
        tick = {"P-TPT": 250.0, "T-TPT": 5.0, "p_hydrate": 0.85,
                "margin_C": -1.0, "sensors": {}, "t": "2024-01-07T08:00"}
        assert should_trigger(tick, well_id="WELL-00019") is True

    def test_should_trigger_false(self):
        tick = {"P-TPT": 280.0, "T-TPT": 85.0, "p_hydrate": 0.05,
                "margin_C": 5.0, "sensors": {}, "t": "2024-01-07T08:00"}
        assert should_trigger(tick, well_id="WELL-TEST-NORMAL") is False

    def test_record_and_recheck(self):
        well = "WELL-RECHECK-TEST"
        now = pd.Timestamp("2024-01-07 08:00:00")
        record_decision(well, "WATCH", now)
        recheck = get_next_recheck(well)
        assert recheck is not None
        assert recheck > now
