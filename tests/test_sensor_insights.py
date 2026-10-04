import numpy as np
import pandas as pd

from src.sensor_insights import SENSOR_INFO, sensor_insights


def frame(**columns):
    index = pd.date_range("2024-01-01", periods=11, freq="min")
    data = {item["code"]: np.full(11, np.nan) for item in SENSOR_INFO}
    data.update({k: np.asarray(v, dtype=float) for k, v in columns.items()})
    return pd.DataFrame(data, index=index)


def ids(result):
    return {item["id"] for item in result["inferences"]}


def test_cooling_with_rising_pressure_flags_hydrate_conditions():
    result = sensor_insights(frame(**{"P-TPT": np.linspace(100, 104, 11), "T-TPT": np.linspace(30, 25, 11)}))
    assert "hydrate_conditions_P-TPT" in ids(result)
    first = result["inferences"][0]
    assert first["level"] == "watch" and "hydrate" in first["why"].lower()


def test_pressure_and_temperature_falling_together_is_flow_loss():
    result = sensor_insights(frame(**{"P-TPT": np.linspace(100, 96, 11), "T-TPT": np.linspace(30, 27, 11)}))
    assert "flow_loss_P-TPT" in ids(result)


def test_growing_line_differential_points_to_line_restriction():
    result = sensor_insights(frame(**{"P-TPT": np.linspace(100, 106, 11), "P-MON-CKP": np.full(11, 50.)}))
    item = next(i for i in result["inferences"] if i["id"] == "line_differential")
    assert item["level"] == "watch" and "restriction in the production line" in item["why"]


def test_choke_movement_explains_pressure_but_fixed_choke_does_not():
    moved = sensor_insights(frame(**{"P-TPT": np.linspace(100, 105, 11), "ABER-CKP": np.linspace(50, 40, 11)}))
    fixed = sensor_insights(frame(**{"P-TPT": np.linspace(100, 105, 11), "ABER-CKP": np.full(11, 50.)}))
    assert "choke_moved" in ids(moved) and "pressure_without_choke" not in ids(moved)
    assert "pressure_without_choke" in ids(fixed)


def test_steady_data_has_no_watch_inferences_and_zero_gauges_are_inactive():
    result = sensor_insights(frame(**{"P-TPT": np.full(11, 100.), "T-TPT": np.full(11, 30.), "P-PDG": np.zeros(11)}))
    assert not [i for i in result["inferences"] if i["level"] == "watch"]
    pdg = next(ch for ch in result["channels"] if ch["code"] == "P-PDG")
    assert pdg["trend"] == "inactive" and "inactive_gauges" in ids(result)


def test_invalid_sentinels_and_missing_channels_are_unavailable():
    result = sensor_insights(frame(**{"P-TPT": np.full(11, 1e30)}))
    tpt = next(ch for ch in result["channels"] if ch["code"] == "P-TPT")
    assert tpt["trend"] == "unavailable" and tpt["value"] is None
