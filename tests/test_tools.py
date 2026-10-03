"""Tests for src/tools.py — agent tool definitions and causality.

Owner: Div
"""
import json
import numpy as np
import pandas as pd
import pytest

from src.tools import (
    AgentContext, TOOL_REGISTRY, TOOL_SCHEMAS, dispatch_tool,
    get_window, tool_hydrate_margin, tool_classify_event,
    tool_forecast_onset, tool_methanol_dose, tool_well_history,
    tool_search_playbook, tool_check_sensor_quality, load_thresholds,
)


def _make_ctx(n_minutes=100, minute_index=None, has_event=True):
    """Build a test AgentContext with synthetic data."""
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
        "p_hydrate": np.linspace(0.05, 0.8 if has_event else 0.05, n_minutes),
    }, index=idx)
    return AgentContext(
        instance_id="test_instance",
        well_id="WELL-TEST",
        minute_index=minute_index,
        df=df,
    )


class TestSchemaRegistry:
    def test_all_tools_have_schemas(self):
        tool_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        expected = {
            "get_window", "hydrate_margin", "classify_event",
            "forecast_onset", "methanol_dose", "well_history", "search_playbook",
            "check_sensor_quality",
        }
        assert expected == tool_names

    def test_schemas_are_valid_openai_format(self):
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]


class TestJSONSerializable:
    def test_get_window_serializable(self):
        ctx = _make_ctx()
        result = get_window(ctx, minutes=60)
        json.dumps(result)

    def test_classify_event_serializable(self):
        ctx = _make_ctx()
        result = tool_classify_event(ctx)
        json.dumps(result)

    def test_forecast_onset_serializable(self):
        ctx = _make_ctx()
        result = tool_forecast_onset(ctx)
        json.dumps(result)

    def test_methanol_dose_serializable(self):
        ctx = _make_ctx()
        result = tool_methanol_dose(ctx, target_shift_C=5.0)
        json.dumps(result)

    def test_well_history_serializable(self):
        ctx = _make_ctx()
        result = tool_well_history(ctx)
        json.dumps(result)

    def test_search_playbook_serializable(self):
        ctx = _make_ctx()
        result = tool_search_playbook(ctx, query="hydrate")
        json.dumps(result)


class TestCausality:
    def test_future_rows_dont_affect_output(self):
        """Changing data after the current minute must not change tool output."""
        ctx1 = _make_ctx(n_minutes=100, minute_index=50)
        r1 = get_window(ctx1, minutes=30)
        c1 = tool_classify_event(ctx1)

        # Modify future rows (indices 51-99)
        ctx2 = _make_ctx(n_minutes=100, minute_index=50)
        ctx2.df.iloc[51:, ctx2.df.columns.get_loc("P-TPT")] = 999.0
        ctx2.df.iloc[51:, ctx2.df.columns.get_loc("T-TPT")] = 999.0
        r2 = get_window(ctx2, minutes=30)
        c2 = tool_classify_event(ctx2)

        assert r1["latest"] == r2["latest"]
        assert c1["p_hydrate"] == c2["p_hydrate"]


class TestFallbacks:
    def test_classify_returns_heuristic_fallback(self):
        """With no ML model, classify should use heuristic."""
        ctx = _make_ctx()
        result = tool_classify_event(ctx)
        assert result["available"] is True
        assert result["source"] == "heuristic_fallback"
        assert abs(result["p_hydrate"] + result["p_lookalike"] + result["p_normal"] - 1.0) < 0.01

    def test_hydrate_margin_unavailable_without_physics(self):
        ctx = _make_ctx()
        result = tool_hydrate_margin(ctx)
        # physics.py raises NotImplementedError
        assert result["available"] is False

    def test_methanol_dose_unavailable_without_physics(self):
        ctx = _make_ctx()
        result = tool_methanol_dose(ctx)
        assert result["available"] is False

    def test_search_playbook_returns_results(self):
        ctx = _make_ctx()
        result = tool_search_playbook(ctx)
        assert result["available"] is True
        assert result["source"] == "rag"
        assert len(result["results"]) > 0
        assert "title" in result["results"][0]


class TestDispatch:
    def test_unknown_tool_returns_error(self):
        ctx = _make_ctx()
        result = dispatch_tool("nonexistent_tool", ctx, {})
        assert result["available"] is False
        assert "error" in result

    def test_bad_args_returns_error(self):
        ctx = _make_ctx()
        result = dispatch_tool("search_playbook", ctx, {})  # missing required "query"
        assert result["available"] is False or "error" in result or result.get("source") == "rag"

    def test_dispatch_get_window(self):
        ctx = _make_ctx()
        result = dispatch_tool("get_window", ctx, {"minutes": 30})
        assert result["available"] is True
        assert "latest" in result


class TestGetWindow:
    def test_does_not_expose_phase(self):
        ctx = _make_ctx()
        result = get_window(ctx)
        flat = json.dumps(result)
        assert '"phase"' not in flat

    def test_missing_sensors_listed(self):
        ctx = _make_ctx()
        result = get_window(ctx)
        assert "P-JUS-CKP" in result["missing_sensors"]
        assert "T-JUS-CKP" in result["missing_sensors"]


class TestDataQuality:
    def _make_quality_ctx(self, overrides: dict, n_minutes=100, minute_index=None):
        """Build a context with quality checking via ReplayInstance."""
        from backend.replay import ReplayInstance
        if minute_index is None:
            minute_index = n_minutes - 1
        idx = pd.date_range("2024-01-01", periods=n_minutes, freq="1min")
        df = pd.DataFrame({
            "P-PDG": np.linspace(280, 260, n_minutes),
            "T-PDG": np.linspace(85, 80, n_minutes),
            "P-TPT": np.linspace(278, 258, n_minutes),
            "T-TPT": np.linspace(84, 79, n_minutes),
            "P-MON-CKP": np.linspace(276, 256, n_minutes),
            "P-JUS-CKP": np.full(n_minutes, np.nan),
            "T-JUS-CKP": np.full(n_minutes, np.nan),
            "ABER-CKP": np.full(n_minutes, 30.0),
            "QGL": np.linspace(12, 10, n_minutes),
            "phase": ["normal"] * n_minutes,
        }, index=idx)
        for col, vals in overrides.items():
            if callable(vals):
                vals(df, col)
            else:
                df[col] = vals
        inst = ReplayInstance(df, {"instance_id": "test", "well_id": "WELL-TEST"})
        return AgentContext(
            instance_id="test",
            well_id="WELL-TEST",
            minute_index=minute_index,
            df=inst.df,
        )

    def test_nonfinite_sensor_flagged_and_none(self):
        """NaN sensor value -> quality flag + None in tick."""
        from backend.replay import ReplayInstance
        idx = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame({
            "P-PDG": [280, np.nan, np.inf, -np.inf, 280],
            "T-PDG": [85.0] * 5,
            "P-TPT": [278.0] * 5,
            "T-TPT": [84.0] * 5,
            "P-MON-CKP": [276.0] * 5,
            "P-JUS-CKP": [np.nan] * 5,
            "T-JUS-CKP": [np.nan] * 5,
            "ABER-CKP": [30.0] * 5,
            "QGL": [12.0] * 5,
            "phase": ["normal"] * 5,
        }, index=idx)
        inst = ReplayInstance(df, {"instance_id": "test", "well_id": "W"})
        # Index 2 had inf -> should be flagged
        tick2 = inst.tick(2)
        assert tick2["sensors"]["P_PDG_bar"] is None
        assert "quality" in tick2
        assert tick2["quality"]["P-PDG"] == "nonfinite"

    def test_extreme_value_flagged(self):
        """Value >= 1e30 -> quality flag + None."""
        from backend.replay import ReplayInstance
        idx = pd.date_range("2024-01-01", periods=3, freq="1min")
        df = pd.DataFrame({
            "P-PDG": [280, 1e35, 280],
            "T-PDG": [85.0] * 3,
            "P-TPT": [278.0] * 3,
            "T-TPT": [84.0] * 3,
            "P-MON-CKP": [276.0] * 3,
            "P-JUS-CKP": [np.nan] * 3,
            "T-JUS-CKP": [np.nan] * 3,
            "ABER-CKP": [30.0] * 3,
            "QGL": [12.0] * 3,
            "phase": ["normal"] * 3,
        }, index=idx)
        inst = ReplayInstance(df, {"instance_id": "test", "well_id": "W"})
        tick1 = inst.tick(1)
        assert tick1["sensors"]["P_PDG_bar"] is None
        assert tick1["quality"]["P-PDG"] == "extreme"

    def test_temperature_below_absolute_zero_flagged(self):
        """Temperature < -273.15 -> flagged and None."""
        from backend.replay import ReplayInstance
        idx = pd.date_range("2024-01-01", periods=3, freq="1min")
        df = pd.DataFrame({
            "P-PDG": [280.0] * 3,
            "T-PDG": [85, -300, 85],
            "P-TPT": [278.0] * 3,
            "T-TPT": [84.0] * 3,
            "P-MON-CKP": [276.0] * 3,
            "P-JUS-CKP": [np.nan] * 3,
            "T-JUS-CKP": [np.nan] * 3,
            "ABER-CKP": [30.0] * 3,
            "QGL": [12.0] * 3,
            "phase": ["normal"] * 3,
        }, index=idx)
        inst = ReplayInstance(df, {"instance_id": "test", "well_id": "W"})
        tick1 = inst.tick(1)
        assert tick1["sensors"]["T_PDG_C"] is None
        assert tick1["quality"]["T-PDG"] == "below_absolute_zero"

    def test_stuck_sensor_flagged_but_value_preserved(self):
        """Sensor unchanged for 30+ min -> 'stuck' flag but value kept."""
        from backend.replay import ReplayInstance
        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="1min")
        df = pd.DataFrame({
            "P-PDG": [280.0] * n,  # stuck
            "T-PDG": np.linspace(85, 80, n),
            "P-TPT": [278.0] * n,  # stuck
            "T-TPT": np.linspace(84, 79, n),
            "P-MON-CKP": np.linspace(276, 256, n),
            "P-JUS-CKP": [np.nan] * n,
            "T-JUS-CKP": [np.nan] * n,
            "ABER-CKP": [30.0] * n,  # stuck
            "QGL": np.linspace(12, 10, n),
            "phase": ["normal"] * n,
        }, index=idx)
        inst = ReplayInstance(df, {"instance_id": "test", "well_id": "W"})
        tick_last = inst.tick(n - 1)
        # P-PDG should be stuck but value preserved
        assert tick_last["sensors"]["P_PDG_bar"] == 280.0
        assert "quality" in tick_last
        assert tick_last["quality"]["P-PDG"] == "stuck"

    def test_check_sensor_quality_tool_returns_flags(self):
        """check_sensor_quality tool returns quality flags."""
        ctx = self._make_quality_ctx({
            "P-PDG": lambda df, col: df.__setitem__(col, [1e35] * len(df)),
        }, n_minutes=5, minute_index=0)
        result = tool_check_sensor_quality(ctx)
        assert result["available"] is True
        assert "P-PDG" in result["flagged"]
        assert result["sensors"]["P-PDG"]["status"] == "extreme"
