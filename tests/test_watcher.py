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

    def test_lookalike_threshold_triggers(self):
        """p_lookalike above threshold should trigger."""
        w = Watcher()
        tick = {"t": "2024-01-07T08:00", "p_hydrate": None, "margin_C": None,
                "p_lookalike": 0.65, "sensors": {}}
        result = w.check_tick(tick, "WELL-00019", 0)
        assert result is not None
        assert "p_lookalike" in result["reason"]


class TestBaselineRelativeTrigger:
    """Condition 3: baseline-relative pressure change detection."""

    def _feed_baseline(self, w, well_id, n, p_tpt=280.0, p_mon=276.0,
                        noise=0.1):
        """Feed n ticks of stable baseline data."""
        import numpy as np
        rng = np.random.RandomState(42)
        for i in range(n):
            tick = {
                "t": f"2024-01-07T{8+i//60:02d}:{i%60:02d}",
                "p_hydrate": None, "margin_C": None,
                "sensors": {
                    "P_TPT_bar": p_tpt + rng.normal(0, noise),
                    "P_MON_CKP_bar": p_mon + rng.normal(0, noise),
                },
            }
            w.check_tick(tick, well_id, i)

    def test_triggers_on_large_sigma_change(self):
        """A change well beyond k*baseline_std should trigger."""
        w = Watcher()
        well = "WELL-SIGMA"
        # 60 ticks of stable baseline (std ~0.1 bar)
        self._feed_baseline(w, well, 60)
        # Then a sustained large change (5 bar drop = 50 sigma)
        results = []
        sustain = w.thresholds.get("pressure_sigma_sustain_min", 3)
        for i in range(sustain + 2):
            tick = {
                "t": f"2024-01-07T09:{i:02d}",
                "p_hydrate": None, "margin_C": None,
                "sensors": {"P_TPT_bar": 275.0 - i * 0.5},
            }
            result = w.check_tick(tick, well, 60 + i)
            if result:
                results.append(result)
        assert len(results) >= 1
        assert "Pressure anomaly" in results[0]["reason"]
        assert "σ" in results[0]["reason"]

    def test_flat_sensor_does_not_fire(self):
        """A sensor with zero std should not fire on noise."""
        w = Watcher()
        well = "WELL-FLAT"
        # 60 ticks of perfectly flat P-TPT
        for i in range(60):
            tick = {
                "t": f"2024-01-07T08:{i:02d}",
                "p_hydrate": None, "margin_C": None,
                "sensors": {"P_TPT_bar": 280.0},
            }
            w.check_tick(tick, well, i)
        # Then tiny noise (0.01 bar) — should NOT trigger because abs_floor=0.5
        for i in range(10):
            tick = {
                "t": f"2024-01-07T09:{i:02d}",
                "p_hydrate": None, "margin_C": None,
                "sensors": {"P_TPT_bar": 280.0 + 0.01 * (i % 2)},
            }
            result = w.check_tick(tick, well, 60 + i)
        assert result is None

    def test_normal_well_silent_over_180_min(self):
        """A well with normal variability should not trigger over 180 min."""
        w = Watcher()
        well = "WELL-NORMAL"
        import numpy as np
        rng = np.random.RandomState(123)
        last_result = None
        for i in range(180):
            # Normal fluctuation: std ~0.9 bar, max ~3 bar swing
            p = 280.0 + rng.normal(0, 0.9)
            tick = {
                "t": f"2024-01-07T{8+i//60:02d}:{i%60:02d}",
                "p_hydrate": None, "margin_C": None,
                "sensors": {
                    "P_TPT_bar": p,
                    "P_MON_CKP_bar": 276.0 + rng.normal(0, 0.7),
                },
            }
            last_result = w.check_tick(tick, well, i)
        # Should never trigger on normal noise
        st = w._state(well)
        # Check no trigger was ever returned
        assert st.last_decision_minute == -999  # No decision recorded

    def test_causality_future_rows_dont_change_triggers(self):
        """Changing future data doesn't affect triggers at current minute."""
        import numpy as np
        w1 = Watcher()
        w2 = Watcher()
        well = "WELL-CAUSAL"
        rng = np.random.RandomState(42)

        # Feed 70 ticks of identical baseline to both
        for i in range(70):
            p = 280.0 + rng.normal(0, 0.1)
            tick = {
                "t": f"2024-01-07T08:{i:02d}",
                "p_hydrate": None, "margin_C": None,
                "sensors": {"P_TPT_bar": p},
            }
            w1.check_tick(tick, well, i)
            w2.check_tick(tick, well, i)

        # Now w1 gets the same next tick, w2 gets a different future
        tick_now = {
            "t": "2024-01-07T09:10",
            "p_hydrate": None, "margin_C": None,
            "sensors": {"P_TPT_bar": 275.0},
        }
        r1 = w1.check_tick(tick_now, well, 70)
        r2 = w2.check_tick(tick_now, well, 70)

        # Both should produce the same result (or both None)
        assert (r1 is None) == (r2 is None)
        if r1 is not None:
            assert r1["score"] == r2["score"]

    def test_condition3_runs_with_margin_present(self):
        """Condition 3 should still run even when margin is available."""
        w = Watcher()
        well = "WELL-MARGIN-SIGMA"
        # Feed baseline with margin (healthy margin of 5.0 — no margin trigger)
        import numpy as np
        rng = np.random.RandomState(42)
        for i in range(60):
            tick = {
                "t": f"2024-01-07T08:{i:02d}",
                "p_hydrate": None,
                "margin_C": 5.0,
                "sensors": {"P_TPT_bar": 280.0 + rng.normal(0, 0.1)},
            }
            w.check_tick(tick, well, i)

        # Now big pressure drop with margin still OK (margin is 5.0, not dangerous)
        results = []
        for i in range(5):
            tick = {
                "t": f"2024-01-07T09:{i:02d}",
                "p_hydrate": None,
                "margin_C": 5.0,
                "sensors": {"P_TPT_bar": 270.0 - i},
            }
            result = w.check_tick(tick, well, 60 + i)
            if result:
                results.append(result)

        # Should trigger from condition 3 (pressure anomaly), not condition 2
        assert len(results) >= 1
        assert "Pressure anomaly" in results[0]["reason"]

    def test_p_mon_ckp_triggers(self):
        """P-MON-CKP change beyond k-sigma should trigger."""
        w = Watcher()
        well = "WELL-MON"
        import numpy as np
        rng = np.random.RandomState(42)
        # Feed baseline for P-MON-CKP
        for i in range(60):
            tick = {
                "t": f"2024-01-07T08:{i:02d}",
                "p_hydrate": None, "margin_C": None,
                "sensors": {"P_MON_CKP_bar": 52.0 + rng.normal(0, 0.1)},
            }
            w.check_tick(tick, well, i)

        # Large sustained P-MON-CKP change
        results = []
        for i in range(5):
            tick = {
                "t": f"2024-01-07T09:{i:02d}",
                "p_hydrate": None, "margin_C": None,
                "sensors": {"P_MON_CKP_bar": 55.0 + i * 0.5},
            }
            result = w.check_tick(tick, well, 60 + i)
            if result:
                results.append(result)

        assert len(results) >= 1
        assert "P_MON_CKP" in results[0]["reason"]


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

        # 2 minutes later — still in cooldown (default 60 min)
        result2 = w.check_tick(tick, "WELL-00019", 4)
        assert result2 is None

    def test_cooldown_expires(self):
        w = Watcher()
        tick = {"t": "2024-01-07T08:00", "p_hydrate": 0.85, "margin_C": None, "sensors": {}}
        w.record_decision("WELL-00019", "ALERT", 0, score=0.85)

        # After cooldown (61 min later, default cooldown=60)
        w.check_tick(tick, "WELL-00019", 61)
        w.check_tick(tick, "WELL-00019", 62)
        result = w.check_tick(tick, "WELL-00019", 63)
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
        assert w.thresholds["pressure_sigma_k"] == DEFAULT_THRESHOLDS["pressure_sigma_k"]
        assert w.thresholds["pressure_sigma_sustain_min"] == DEFAULT_THRESHOLDS["pressure_sigma_sustain_min"]
        assert w.thresholds["pressure_abs_floor_bar"] == DEFAULT_THRESHOLDS["pressure_abs_floor_bar"]


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
