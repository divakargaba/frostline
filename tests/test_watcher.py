"""Tests for src/watcher.py — per-tick anomaly trigger.

Owner: Div
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.pending


class TestTrigger:
    def test_triggers_above_threshold(self):
        """Watcher should trigger when anomaly score exceeds threshold."""
        from src.watcher import should_trigger

        tick = {"P-TPT": 250.0, "T-TPT": 5.0, "p_hydrate": 0.85}
        assert should_trigger(tick, well_id="WELL-00019") is True

    def test_no_trigger_normal(self):
        """Watcher should not trigger during clearly normal conditions."""
        from src.watcher import should_trigger

        tick = {"P-TPT": 280.0, "T-TPT": 85.0, "p_hydrate": 0.05}
        assert should_trigger(tick, well_id="WELL-00019") is False


class TestCooldown:
    def test_cooldown_prevents_retrigger(self):
        """After a decision, watcher must not retrigger within cooldown window."""
        from src.watcher import record_decision, should_trigger

        well = "WELL-00019"
        now = pd.Timestamp("2024-01-07 08:00:00")
        record_decision(well, "ALERT", now)

        # 2 minutes later — still in cooldown
        tick = {"P-TPT": 250.0, "T-TPT": 5.0, "p_hydrate": 0.85}
        # should_trigger should check cooldown and return False
        # (exact cooldown window is implementation detail)
        result = should_trigger(tick, well)
        assert result is False


class TestRecheck:
    def test_recheck_after_watch(self):
        """A WATCH decision should schedule a recheck."""
        from src.watcher import get_next_recheck, record_decision

        well = "WELL-00019"
        now = pd.Timestamp("2024-01-07 08:00:00")
        record_decision(well, "WATCH", now)

        recheck = get_next_recheck(well)
        assert recheck is not None
        assert recheck > now
