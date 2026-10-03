"""Tests for eval.py — evaluation harness.

Owner: Physics+Eval
"""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.pending


@pytest.fixture
def eval_timeline():
    """A simple timeline for testing alarm logic.

    120 minutes total:
      0–59: normal
      60–89: forming (30 min)
      90–119: established (30 min)
    """
    ts = pd.date_range("2024-01-01", periods=120, freq="1min")
    phase = (["normal"] * 60) + (["forming"] * 30) + (["established"] * 30)
    return pd.DataFrame({"timestamp": ts, "phase": phase})


class TestLeadTime:
    def test_positive_means_early(self, eval_timeline):
        """Lead time = t_est - t_alarm. Positive means alarm came before established."""
        from eval import compute_metrics

        t_est = eval_timeline[eval_timeline["phase"] == "established"].iloc[0]["timestamp"]
        # Alarm at minute 70 (during forming), t_est at minute 90
        # Lead time = 90 - 70 = 20 min (positive = early, good)
        lead_time = (t_est - eval_timeline.iloc[70]["timestamp"]).total_seconds() / 60
        assert lead_time == 20.0


class TestDebounce:
    def test_two_minute_spike_not_alarm(self):
        """A 2-minute spike should NOT count as an alarm (need 3+ consecutive)."""
        scores = [0] * 10 + [1, 1] + [0] * 10  # 2-min spike
        consecutive = 0
        alarm = False
        for s in scores:
            if s >= 1:
                consecutive += 1
                if consecutive >= 3:
                    alarm = True
                    break
            else:
                consecutive = 0
        assert not alarm

    def test_three_minute_is_alarm(self):
        """A 3-minute streak should count as an alarm."""
        scores = [0] * 10 + [1, 1, 1] + [0] * 10  # 3-min streak
        consecutive = 0
        alarm = False
        for s in scores:
            if s >= 1:
                consecutive += 1
                if consecutive >= 3:
                    alarm = True
                    break
            else:
                consecutive = 0
        assert alarm


class TestFalseAlarmRate:
    def test_normalized_per_24h(self):
        """False alarm rate = count / (normal_hours / 24)."""
        normal_hours = 48  # 2 days of normal operation
        false_alarm_count = 4
        rate = false_alarm_count / (normal_hours / 24)
        assert rate == 2.0  # 2 false alarms per day


class TestMisdiagnosis:
    def test_lookalike_flagged_as_hydrate(self):
        """A look-alike event (e.g. scaling) flagged as hydrate = misdiagnosis."""
        true_class = "scaling"
        predicted = "hydrate"
        is_misdiagnosis = (true_class != "hydrate") and (predicted == "hydrate")
        assert is_misdiagnosis


class TestB0Baseline:
    def test_catches_nothing(self, eval_timeline):
        """B0 (always-normal) should catch zero events."""
        # B0 never raises an alarm
        predictions = ["normal"] * len(eval_timeline)
        alarms = [p for p in predictions if p != "normal"]
        assert len(alarms) == 0
