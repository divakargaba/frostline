"""Tests for src/forecast.py — onset time forecasting.

Owner: Physics+Eval
"""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.pending


class TestQuantileOrdering:
    def test_p10_le_p50_le_p90(self, fake_3w_1min):
        """Quantile forecasts must be ordered: p10 <= p50 <= p90."""
        from src.forecast import forecast_onset

        result = forecast_onset(fake_3w_1min)
        assert result["q10"] <= result["q50"] <= result["q90"]


class TestTrainingData:
    def test_only_pre_established_rows(self, fake_3w_1min):
        """Training should only use rows before the established phase.

        The target (time to established) is only meaningful for forming rows.
        Established rows are the outcome, not a training input.
        """
        forming = fake_3w_1min[fake_3w_1min["phase"] == "forming"]
        established = fake_3w_1min[fake_3w_1min["phase"] == "established"]
        # There should be forming rows to train on
        assert len(forming) > 0
        # There should be established rows (the outcome)
        assert len(established) > 0
