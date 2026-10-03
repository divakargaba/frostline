import json
import numpy as np
import pandas as pd
import pytest
from src.load import SENSORS
from src.real_pilot import OUT, alarms, causal_features, event_metrics, minute_frame


def test_features_ignore_future_values_and_all_labels():
    frame = pd.DataFrame({s: np.arange(90, dtype=float) for s in SENSORS}, index=pd.date_range("2024-01-01", periods=90, freq="min"))
    frame["class"] = 0
    before = causal_features(frame)
    frame.iloc[60:, :len(SENSORS)] = -10000
    frame["class"] = 108
    after = causal_features(frame)
    pd.testing.assert_frame_equal(before.iloc[:60], after.iloc[:60])
    assert not any(c in after.columns for c in ["class", "state", "well_id", "phase"])


def test_minute_aggregation_is_timestamped_at_end_and_keeps_frozen(tmp_path):
    raw = pd.DataFrame({"P-TPT": [20000000] * 120, "T-TPT": [42] * 120, "class": [0] * 120}, index=pd.date_range("2024-01-01", periods=120, freq="s"))
    path = tmp_path / "recording.parquet"
    raw.to_parquet(path)
    frame = minute_frame(path)
    assert frame.index[0] == pd.Timestamp("2024-01-01 00:01")
    assert frame["P-TPT"].tolist() == [200, 200]
    assert frame["P-MON-CKP"].isna().all()


def test_real_alarm_requires_three_available_consecutive_minutes():
    frame = pd.DataFrame({s: [100.] * 7 for s in SENSORS}, index=pd.date_range("2024-01-01", periods=7, freq="min"))
    frame.loc[frame.index[3], ["P-PDG", "P-TPT", "P-MON-CKP"]] = np.nan
    assert alarms([.9] * 7, frame).tolist() == [False, False, True, False, False, False, True]


def test_early_vs_late_lead_time_is_measured_against_established():
    frame = pd.DataFrame({s: [100.] * 7 for s in SENSORS}, index=pd.date_range("2024-01-01", periods=7, freq="min"))
    frame["class"] = [0, 108, 108, 108, 8, 8, 8]
    p = np.tile([0, 1, 0], (7, 1))
    result = event_metrics(frame, p.argmax(axis=1), p)
    assert result["early"]
    assert result["lead_minutes"] == 2
    p[:3] = [1, 0, 0]
    result = event_metrics(frame, p.argmax(axis=1), p)
    assert result["late"]
    assert result["lead_minutes"] == -1


@pytest.mark.skipif(not (OUT / "real_report.json").exists(), reason="Download and run the optional real-data pilot first")
def test_saved_pilot_has_disjoint_wells_and_one_heldout_fold_per_recording():
    report = json.loads((OUT / "real_report.json").read_text())
    test_wells = []
    for fold in report["folds"]:
        assert not set(fold["train_wells"]) & set(fold["test_wells"])
        test_wells.extend(fold["test_wells"])
        assert all(r["well"] in fold["test_wells"] for r in report["recordings"] if r["fold"] == fold["fold"])
    assert len(set(test_wells)) == len(test_wells) == report["wells"]
    assert sum(map(sum, report["confusion_matrix"])) == report["minutes"]
