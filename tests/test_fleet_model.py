import json

import numpy as np
import pandas as pd
import pytest

from src.fleet_model import (
    BUNDLE_PATH, CLASS_NAMES, DEFAULT_POLICY, EXCLUDED_WELLS, REPORT_PATH,
    SENSORS, advance_alarm, aggregate_metrics, candidate_qualifies, infer_window, load_bundle,
    prepare_recording, recording_metrics,
)
from src.real_pilot import causal_features


class SensorModel:
    def predict_proba(self, frame):
        pressure = frame["P-TPT"].iloc[-1]
        hydrate = .8 if pressure > 110 else .2
        return np.array([[1 - hydrate, hydrate, 0.]])


def sensor_frame(n=90):
    frame = pd.DataFrame({name: np.linspace(100., 120., n) for name in SENSORS},
                         index=pd.date_range("2024-01-01", periods=n, freq="min"))
    frame["class"] = 0
    return frame


def small_bundle(frame):
    return {"model": SensorModel(), "feature_names": list(causal_features(frame).columns), "model_id": "test"}


def test_preparation_retains_unknowns_gaps_and_only_masks_invalid_values(tmp_path):
    index = pd.date_range("2024-01-01", periods=240, freq="s")
    raw = pd.DataFrame({"P-TPT": 2e7, "T-TPT": 42., "P-PDG": -1.18e42,
                        "QGL": 0., "ABER-CKP": 50., "class": 0.}, index=index)
    raw.loc[index[:60], "class"] = np.nan
    raw.loc[index[120:180], "T-TPT"] = -300
    raw.loc[index[180:], "ABER-CKP"] = 101
    raw = raw.drop(index[60:120])
    path = tmp_path / "raw.parquet"
    raw.to_parquet(path)
    before = path.read_bytes()
    frame = prepare_recording(path)
    assert path.read_bytes() == before
    assert len(frame) == 4
    assert frame.index[0] == pd.Timestamp("2024-01-01 00:01")
    assert frame["class"].iloc[:2].isna().all()
    assert frame["P-PDG"].isna().all()
    assert frame["P-TPT"].iloc[0] == 200
    assert pd.isna(frame["T-TPT"].iloc[2])
    assert frame["T-TPT"].iloc[3] == 42.
    assert pd.isna(frame["ABER-CKP"].iloc[3])
    assert frame["QGL"].dropna().eq(0).all()
    assert frame["invalid_P-PDG"].iloc[0] == 1


def test_inference_ignores_labels_and_future_values():
    frame = sensor_frame()
    bundle = small_bundle(frame)
    expected = infer_window(frame.iloc[:60], bundle)
    poisoned = frame.copy()
    poisoned["class"] = 108
    poisoned["state"] = 8
    poisoned.iloc[60:, :len(SENSORS)] = -1e20
    assert infer_window(poisoned.iloc[:60], bundle) == expected
    assert not any(name in bundle["feature_names"] for name in ["class", "state"])
    assert infer_window(frame, bundle)["scores"]["hydrate"] == .8


def test_zero_flow_and_constant_opening_do_not_create_sensor_fault():
    frame = sensor_frame()
    frame["QGL"] = 0.
    frame["ABER-CKP"] = 50.
    result = infer_window(frame, small_bundle(frame))
    assert "QGL" not in result["quality"]["invalid"]
    assert "QGL" not in result["quality"]["unchanged"]
    assert "ABER-CKP" not in result["quality"]["unchanged"]
    assert not result["quality"]["blocked"]


def test_no_pressure_abstains_and_json_remains_finite():
    frame = sensor_frame()
    frame.loc[frame.index[-1], ["P-PDG", "P-TPT", "P-MON-CKP"]] = np.nan
    result = infer_window(frame, small_bundle(frame))
    assert result["quality"]["blocked"]
    assert result["scores"] == {name: None for name in CLASS_NAMES}
    json.dumps(result, allow_nan=False)


def test_live_invalid_injection_is_sanitized_before_model_inference():
    frame = sensor_frame()
    frame.loc[frame.index[-1], ["P-PDG", "P-TPT", "P-MON-CKP"]] = -1e35
    result = infer_window(frame, small_bundle(frame))
    assert result["quality"]["blocked"]
    assert set(result["quality"]["invalid"]) >= {"P-PDG", "P-TPT", "P-MON-CKP"}


def test_alarm_activation_recovery_and_missing_evidence():
    state = {}
    clock = pd.date_range("2024-01-01", periods=12, freq="min")
    for i in range(3):
        state = advance_alarm(state, .8, True, clock[i])
    assert state["active"] and state["triggered"]
    state = advance_alarm(state, None, False, clock[3])
    assert state["active"] and state["suspended"] and not state["alarm"]
    assert not state["recovered"]
    for i in range(4, 8):
        state = advance_alarm(state, .2, True, clock[i])
        assert state["active"]
    state = advance_alarm(state, .2, True, clock[8])
    assert not state["active"] and state["recovered"]
    with pytest.raises(ValueError):
        advance_alarm(state, .8, True, clock[8])


def test_gap_resets_activation_persistence():
    state = advance_alarm({}, .8, True, "2024-01-01 00:00")
    state = advance_alarm(state, .8, True, "2024-01-01 00:01")
    state = advance_alarm(state, .8, True, "2024-01-01 00:03")
    assert state["activation_streak"] == 1 and not state["active"]


def test_false_alarm_duration_distinct_from_raw_argmax_confusion():
    frame = sensor_frame(10)
    # Raw argmax says hydrate, but score below .5 never activates an alarm.
    probabilities = np.tile([.3, .4, .3], (10, 1))
    result = recording_metrics(frame, probabilities)
    assert result["classification_confusion"][0][1] == 10
    assert result["false_alarm_minutes"] == 0
    probabilities[:] = [.1, .8, .1]
    result = recording_metrics(frame, probabilities)
    assert result["false_alarm_minutes"] == 8
    assert result["false_alarm_episodes"] == 1
    assert aggregate_metrics([result])["false_alarm_minutes_per_normal_day"] == 1152


def test_candidate_cannot_trade_more_false_alarm_minutes_for_more_detected_events():
    incumbent = {"events_detected": 3, "false_alarm_minutes": 100}
    assert not candidate_qualifies({"events_detected": 4, "false_alarm_minutes": 101}, incumbent)
    assert not candidate_qualifies({"events_detected": 2, "false_alarm_minutes": 0}, incumbent)
    assert not candidate_qualifies(dict(incumbent), incumbent)
    assert candidate_qualifies({"events_detected": 3, "false_alarm_minutes": 99}, incumbent)
    assert candidate_qualifies({"events_detected": 4, "false_alarm_minutes": 100}, incumbent)


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="Train fleet model first")
def test_saved_bundle_and_report_exclude_every_demo_well():
    bundle = load_bundle()
    report = json.loads(REPORT_PATH.read_text())
    assert len(bundle["training_wells"]) == 17
    assert len(bundle["training_recordings"]) == 20
    assert set(bundle["excluded_wells"]) == set(EXCLUDED_WELLS)
    assert not set(bundle["training_wells"]) & set(EXCLUDED_WELLS)
    for fold in report["folds"]:
        assert not set(fold["train_wells"]) & set(fold["validation_wells"])
        assert not (set(fold["train_wells"]) | set(fold["validation_wells"])) & set(EXCLUDED_WELLS)
    assert len(report["candidates"]) == 15
    assert report["evaluation_recordings"] == 11
    assert report["validation_selected"]["events_detected"] >= report["validation_baseline"]["events_detected"]
    assert report["validation_selected"]["false_alarm_minutes"] <= report["validation_baseline"]["false_alarm_minutes"]
    assert report["evaluation_hydrate_wells"] == ["WELL-00019"]
