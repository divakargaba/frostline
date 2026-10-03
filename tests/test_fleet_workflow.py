"""Measured-workflow evaluation parity and metric semantics."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_fleet_workflow import (
    _score_outputs, aggregate, batch_inference_results, evaluate_recording,
    synthetic_lifecycle_checks, verify_streaming_parity,
)
from src.fleet_model import BUNDLE_PATH, DEFAULT_POLICY, SENSORS, feature_frame, load_bundle


class VectorSensorModel:
    def predict_proba(self, values):
        hydrate = np.where(values["P-TPT"].to_numpy() > 110, .8, .05)
        return np.column_stack([1 - hydrate - .05, hydrate, np.full(len(values), .05)])


def data(n=240):
    result = pd.DataFrame({name: np.linspace(100, 120, n) for name in SENSORS},
                          index=pd.date_range("2024-01-01", periods=n, freq="min"))
    result["ABER-CKP"] = 50.
    result["QGL"] = 0.
    result["class"] = 0.
    for name in SENSORS:
        result[f"invalid_{name}"] = 0.
    return result


def bundle(frame):
    return {"model": VectorSensorModel(), "feature_names": list(feature_frame(frame).columns),
            "model_id": "synthetic-vector-model", "feature_family": "base", "policy": DEFAULT_POLICY}


def test_batched_quality_evidence_scores_match_every_stream_prefix():
    frame = data(200)
    frame.loc[frame.index[20:27], ["P-TPT", "P-PDG", "P-MON-CKP"]] = np.nan
    frame.loc[frame.index[34], "P-PDG"] = -1e40
    frame.loc[frame.index[36], "T-TPT"] = -300
    frame.loc[frame.index[40:150], "P-MON-CKP"] = 90.
    frame.loc[frame.index[50], "invalid_P-PDG"] = .2
    frozen = bundle(frame)
    results = list(batch_inference_results(frame, frozen))
    assert verify_streaming_parity(frame, frozen, results, range(len(frame))) == len(frame)
    assert results[22]["quality"]["blocked"]
    assert results[34]["quality"]["invalid"] == ["P-PDG"]
    assert "P-MON-CKP" in results[120]["quality"]["unchanged"]
    assert "QGL" not in results[120]["quality"]["unchanged"]


@pytest.mark.skipif(not BUNDLE_PATH.exists(), reason="Train the frozen fleet model first")
def test_saved_model_batched_scores_match_streaming_with_gaps_and_constant_channels():
    frame = data()
    frame.loc[frame.index[20:30], "P-PDG"] = np.nan
    frame.loc[frame.index[62:68], "P-TPT"] = np.nan
    frame.loc[frame.index[150:], "P-MON-CKP"] = 80.
    frozen = load_bundle()
    predictions = list(batch_inference_results(frame, frozen))
    assert verify_streaming_parity(frame, frozen, predictions, [0, 20, 29, 59, 60, 62, 67, 100, 179, 180, 230, 239]) == 12


def test_future_and_labels_do_not_affect_batched_current_scores_or_policy_decisions():
    frame = data(100)
    frozen = bundle(frame)
    expected = list(batch_inference_results(frame, frozen))
    poisoned = frame.copy()
    poisoned["class"] = 108
    poisoned["ground_truth"] = "attention"
    poisoned.iloc[50:, :len(SENSORS)] = 1e30
    actual = list(batch_inference_results(poisoned, frozen))
    assert actual[:50] == expected[:50]
    score1 = evaluate_recording(frame.iloc[:50], frozen)
    score2 = evaluate_recording(poisoned.iloc[:50], frozen)
    assert score1["assessment_minutes_by_scenario"] == score2["assessment_minutes_by_scenario"]
    assert score1["workflow"]["review_episodes"] == score2["workflow"]["review_episodes"]
    assert score1["workflow"]["events"] != score2["workflow"]["events"]


def test_screening_coverage_does_not_count_missing_data_as_a_process_detection():
    index = pd.date_range("2024-01-01", periods=7, freq="min")
    labels = [0, 108, 8, 0, 106, 6, np.nan]
    scores = _score_outputs(labels, index, ["watch", "watch", "attention", "normal", "unavailable", "unavailable", "watch"])
    assert scores["events"]["hydrate"]["flagged"]
    assert scores["events"]["hydrate"]["preexisting_at_onset"]
    assert scores["events"]["hydrate"]["first_flag_delay_minutes"] == 0
    assert scores["events"]["hydrate"]["new_flag_delay_minutes"] is None
    assert not scores["events"]["restriction"]["flagged"]
    assert scores["false_watch_minutes"] == 1
    assert scores["false_attention_minutes"] == 0
    assert scores["unavailable_minutes"] == 2
    assert scores["labelled_minutes"] == 6


def test_new_event_delay_and_aggregate_are_recording_level_not_minute_counts():
    index = pd.date_range("2024-01-01", periods=6, freq="min")
    scores = _score_outputs([0, 0, 107, 7, 7, 7], index, ["normal", "normal", "normal", "normal", "watch", "attention"])
    assert scores["events"]["scaling"]["first_flag_delay_minutes"] == 2
    assert scores["events"]["scaling"]["new_flag_delay_minutes"] == 2
    records = [{"workflow": scores}, {"workflow": deepcopy(scores)}]
    total = aggregate(records, "workflow")
    assert total["events"]["scaling"]["recordings"] == 2
    assert total["events"]["scaling"]["recordings_flagged"] == 2
    assert total["events"]["scaling"]["event_minutes"] == 8
    assert total["events"]["scaling"]["flagged_minutes"] == 4
    assert total["events"]["scaling"]["mean_first_flag_delay_minutes"] == 2


def test_parity_checker_fails_on_drift_instead_of_publishing_scores():
    frame = data(10)
    frozen = bundle(frame)
    results = list(batch_inference_results(frame, frozen))
    results[4]["scores"]["hydrate"] = .99
    with pytest.raises(AssertionError, match="Score batch/stream mismatch"):
        verify_streaming_parity(frame, frozen, results, [4])


def test_synthetic_lifecycle_cases_are_executed_and_explicitly_separate():
    cases = synthetic_lifecycle_checks()
    assert len(cases) == 6
    assert all(case["status"] == "pass" and case["synthetic"] for case in cases)
    assert len({case["id"] for case in cases}) == len(cases)
