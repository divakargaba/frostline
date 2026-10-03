"""Tests for src/features.py — feature engineering.

Owner: Data+ML
"""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.pending


class TestCausality:
    def test_future_change_does_not_affect_past(self, fake_3w_1min):
        """Changing a future row must not change features of past rows."""
        from src.features import build_features

        df1 = build_features(fake_3w_1min.copy())

        df_mod = fake_3w_1min.copy()
        # Change the last row's pressure to something extreme
        df_mod.iloc[-1, df_mod.columns.get_loc("P-PDG")] = 999.0
        df2 = build_features(df_mod)

        # All rows except the last should be identical
        if len(df1) > 1:
            pd.testing.assert_frame_equal(
                df1.iloc[:-1].filter(like="mean"),
                df2.iloc[:-1].filter(like="mean"),
            )


class TestWindowLengths:
    def test_rolling_columns_exist(self, fake_3w_1min):
        """build_features should produce columns for 10, 30, 60-min windows."""
        from src.features import build_features

        df = build_features(fake_3w_1min)
        for w in [10, 30, 60]:
            matching = [c for c in df.columns if str(w) in c]
            assert len(matching) > 0, f"No columns found for {w}-min window"


class TestNaNTolerance:
    def test_missing_sensor_produces_nan_features(self, fake_3w_1min):
        """When T-JUS-CKP is all NaN, its features should be NaN too."""
        from src.features import build_features

        df = build_features(fake_3w_1min)
        # T-JUS-CKP features (if computed) should be NaN
        tjus_cols = [c for c in df.columns if "T-JUS-CKP" in c or "T_JUS_CKP" in c]
        for col in tjus_cols:
            assert df[col].isna().all(), f"{col} should be all NaN for missing sensor"
