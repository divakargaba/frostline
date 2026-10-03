"""Tests for src/features.py — feature engineering.

Owner: Data+ML
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    MARGIN,
    build_features,
    feature_columns,
    pressure_differentials,
)
from src.load import SENSORS


def _ramp(n: int = 240, instance_id: str = "WELL-00099_a", start="2024-01-01") -> pd.DataFrame:
    """n minutes of processed-style data; P-TPT falls 0.5 bar/min, others noisy."""
    rng = np.random.RandomState(0)
    idx = pd.date_range(start, periods=n, freq="1min", name="timestamp")
    df = pd.DataFrame({s: np.nan for s in SENSORS}, index=idx)
    df["P-PDG"] = 300 + rng.normal(0, 1, n)
    df["P-TPT"] = 280 - 0.5 * np.arange(n)
    df["T-TPT"] = 80 + rng.normal(0, 0.5, n)
    df["P-MON-CKP"] = 270 + rng.normal(0, 1, n)
    df["class"] = 0
    df["instance_id"] = instance_id
    return df


class TestCausality:
    def test_future_change_does_not_affect_past(self, fake_3w_1min):
        """Changing a future row must not change features of past rows."""
        df1 = build_features(fake_3w_1min.copy())
        df_mod = fake_3w_1min.copy()
        df_mod.iloc[-1, df_mod.columns.get_loc("P-PDG")] = 999.0
        df2 = build_features(df_mod)
        cols = feature_columns(df1)
        pd.testing.assert_frame_equal(df1.iloc[:-1][cols], df2.iloc[:-1][cols])

    def test_future_change_long_series(self):
        """All features of every earlier minute are unchanged by a later spike."""
        base = _ramp()
        df1 = build_features(base.copy())
        mod = base.copy()
        mod.iloc[150:, mod.columns.get_loc("P-TPT")] += 50
        df2 = build_features(mod)
        cols = feature_columns(df1)
        pd.testing.assert_frame_equal(df1.iloc[:150][cols], df2.iloc[:150][cols])
        assert not df1.iloc[150][cols].equals(df2.iloc[150][cols])


class TestWindowLengths:
    def test_rolling_columns_exist(self, fake_3w_1min):
        df = build_features(fake_3w_1min)
        for w in [10, 30, 60]:
            for stat in ["mean", "std", "slope"]:
                assert f"P-PDG_{stat}_{w}" in df.columns
        assert "P-PDG_delta_60" in df.columns and "P-PDG_delta_360" in df.columns

    def test_slope_of_linear_ramp(self):
        df = build_features(_ramp())
        for w in [10, 30, 60]:
            assert df[f"P-TPT_slope_{w}"].iloc[-1] == pytest.approx(-0.5)
            assert df[f"P-TPT_mean_{w}"].iloc[-1] == pytest.approx(
                df["P-TPT"].iloc[-w:].mean())

    def test_min_periods(self):
        """No 60-min stat until half the window has data."""
        df = build_features(_ramp())
        assert df["P-TPT_mean_60"].iloc[:29].isna().all()
        assert df["P-TPT_mean_60"].iloc[29:].notna().all()

    def test_time_gap_respected(self):
        """Windows are time-based: a 60-minute gap empties the 10-min window."""
        df = _ramp(120)
        df = df.drop(df.index[40:100])  # 60-minute hole left by dropped minutes
        out = build_features(df)
        first_after_gap = out.index[40]
        assert pd.isna(out.loc[first_after_gap, "P-TPT_mean_10"])


class TestDifferentials:
    def test_differentials(self, fake_3w_1min):
        df = pressure_differentials(fake_3w_1min)
        np.testing.assert_allclose(
            df["P_TPT_minus_P_MON"], fake_3w_1min["P-TPT"] - fake_3w_1min["P-MON-CKP"])
        np.testing.assert_allclose(
            df["P_PDG_minus_P_TPT"], fake_3w_1min["P-PDG"] - fake_3w_1min["P-TPT"])


class TestNaNTolerance:
    def test_missing_sensor_produces_nan_features(self, fake_3w_1min):
        """When T-JUS-CKP is all NaN, its features should be NaN too."""
        df = build_features(fake_3w_1min)
        tjus_cols = [c for c in df.columns if "T-JUS-CKP" in c]
        assert tjus_cols
        for col in tjus_cols:
            assert df[col].isna().all(), f"{col} should be all NaN for missing sensor"

    def test_absent_columns_added(self):
        """Sensors absent from the input still get (NaN) feature columns."""
        df = _ramp().drop(columns=["QGL", "ABER-CKP"])
        out = build_features(df)
        assert set(feature_columns()) <= set(out.columns)
        assert out["QGL_mean_10"].isna().all()


class TestPhysicsHook:
    def test_margin_fn_used_and_rolled(self):
        df = build_features(_ramp(), margin_fn=lambda p, t: p / 100 - t / 10)
        expected = df["P-TPT"] / 100 - df["T-TPT"] / 10
        np.testing.assert_allclose(df[MARGIN], expected)
        assert df[f"{MARGIN}_mean_10"].notna().any()

    def test_margin_nan_when_temperature_missing(self):
        df = _ramp()
        df["T-TPT"] = np.nan
        out = build_features(df, margin_fn=lambda p, t: 1.0)
        assert out[MARGIN].isna().all()


class TestInstances:
    def test_windows_do_not_cross_instances(self):
        a = _ramp(120, "A", start="2024-01-01 00:00")
        b = _ramp(120, "B", start="2024-01-01 02:00")  # B starts right after A
        b["P-TPT"] = 1000.0 + np.random.RandomState(1).normal(0, 1, 120)
        both = build_features(pd.concat([a, b]))
        alone = build_features(b.copy())
        cols = feature_columns(alone)
        pd.testing.assert_frame_equal(
            both[both["instance_id"] == "B"][cols], alone[cols])

    def test_feature_columns_exclude_metadata(self):
        cols = feature_columns()
        for bad in ["class", "state", "phase", "instance_id", "well_id", "source", "event_class"]:
            assert bad not in cols
        assert not any(c.startswith("has_") for c in cols)
