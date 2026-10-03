"""Behavioral checks for the implemented research workflow, not placeholder APIs."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from src.research import SENSORS, Policy, experiment, fit_policy, investigate, load_seed, metrics, predict, seed_scenarios, select_policy, split_seed


@pytest.fixture(scope="module")
def report():
    return experiment()


def test_chronological_partition_keeps_missing_hours():
    cal, val, test = split_seed(load_seed())
    assert [len(x) for x in [cal, val, test]] == [240, 240, 240]
    assert cal.timestamp.max() < val.timestamp.min() < test.timestamp.min()
    assert [x.pressure_bar.notna().sum() for x in [cal, val, test]] == [236, 234, 234]
    assert test.label.sum() == 12


def test_official_starter_parity(report):
    # Independent reference calculation: exactly the organizer's dropna/split/q05.
    original = load_seed().dropna(subset=["pressure_bar"])
    cutoff = original.timestamp.min() + pd.Timedelta(days=20)
    threshold = original[original.timestamp < cutoff].pressure_bar.quantile(.05)
    test = original[original.timestamp >= cutoff]
    flagged = test.pressure_bar < threshold
    measured = next(x for x in report["systems"] if x["id"] == "B1")["test"]
    assert measured["true_positive"] == int((flagged & test.label.eq(1)).sum()) == 5
    assert measured["false_positive"] == int((flagged & test.label.eq(0)).sum()) == 0
    assert measured["evaluated_hours"] == 234


def test_final_test_cannot_select_or_fit_policy(report):
    df = load_seed()
    final = df.timestamp >= df.timestamp.min() + pd.Timedelta(days=20)
    df.loc[final, "label"] = 1 - df.loc[final, "label"]
    df.loc[final, SENSORS] = [10000, -10000, 10000]
    changed = experiment(df)
    assert changed["selection"] == report["selection"]
    assert changed["policy_id"] == report["policy_id"]
    assert [s["policy"] for s in changed["systems"]] == [s["policy"] for s in report["systems"]]


def test_forecast_free_prefix_invariance(report):
    _, _, test = split_seed(load_seed())
    policy = Policy(**report["systems"][-1]["policy"])
    sensors = test[["timestamp", *SENSORS]]
    before = investigate(sensors.iloc[:132], policy)
    poisoned = sensors.copy()
    poisoned.loc[132:, SENSORS] = [-99999, -99999, -99999]
    assert investigate(poisoned, policy)[:132] == before
    for smooth in [False, True]:
        p = replace(policy, smoothing=smooth, consecutive=2)
        np.testing.assert_array_equal(predict(sensors.iloc[:132], p), predict(poisoned, p).iloc[:132])


def test_missing_or_gap_breaks_consecutive_evidence():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=6, freq="h"), "pressure_bar": [2, 2, np.nan, 2, 2, 2], "temp_C": [2] * 6, "flow_Lps": [2] * 6})
    p = Policy(10, None, 2, False, 3)
    assert predict(df, p).tolist() == [False, True, False, False, True, True]
    df.loc[5, "timestamp"] += pd.Timedelta(hours=1)
    assert not predict(df, p).iloc[-1]


def test_ewma_does_not_fill_missing_sensor():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=4, freq="h"), "pressure_bar": [2, np.nan, 8, 2], "temp_C": [2] * 4, "flow_Lps": [2] * 4})
    p = Policy(10, None, 1, True, 3)
    assert predict(df, p).tolist() == [True, False, False, False]


def test_all_search_candidates_only_need_historical_windows():
    cal, val, _ = split_seed(load_seed())
    search = select_policy(cal, val)
    assert len(search["trials"]) == 24
    assert sum(t["selected"] for t in search["trials"]) == 1
    assert search["validation_cost"] < search["incumbent_cost"]
    assert search["selected_config"]["pressure_percentile"] == 20
    assert search["selected_config"]["confirmation_percentile"] == 10


def test_selector_rejects_non_improving_candidates():
    cal, val, _ = split_seed(load_seed())
    # No positives and no threshold crossings: every policy ties at zero cost.
    val.loc[:, "label"] = 0
    val.loc[:, SENSORS] = [1000, 1000, 1000]
    search = select_policy(cal, val)
    assert not search["promoted"]
    assert search["selected_trial"] is None
    assert search["selected_config"]["confirmation_percentile"] is None


def test_fault_scenarios_defer_and_dip_is_not_diagnosed(report):
    replays = seed_scenarios(report)
    assert replays["seed-dip"]["frames"][24]["decision"] == "WATCH"
    assert all(f["decision"] == "WATCH" for f in replays["seed-missing"]["frames"][20:29])
    # Freeze started at index 18 using the value at 17; six identical by 22.
    frame = replays["seed-frozen"]["frames"][22]
    assert frame["decision"] == "WATCH"
    assert set(frame["evidence"][0]["result"]["frozen"]) == set(SENSORS)


def test_actual_seed_replay_alerts_match_scored_predictions(report):
    replay = seed_scenarios(report)["seed-test"]
    _, _, test = split_seed(load_seed())
    alerts = [f["decision"] == "ALERT" for f in replay["frames"]]
    assert metrics(test, alerts) == report["systems"][-1]["test"]
    times = [pd.Timestamp(f["t"]) for f in replay["frames"] if f["notify"]]
    assert all((b - a) >= pd.Timedelta(hours=6) for a, b in zip(times, times[1:]))


def test_one_hit_does_not_get_credit_for_an_entire_incident():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="h"), "pressure_bar": [1] * 5, "label": [0, 1, 1, 1, 0]})
    measured = metrics(df, [False, False, True, False, False])
    assert measured["events_detected"] == 1
    assert measured["true_positive"] == 1
    assert measured["false_negative"] == 2
    assert measured["delay_hours"] == 1


def test_false_alarm_episodes_and_missing_hours_are_explicit():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=6, freq="h"), "pressure_bar": [1, 1, 1, np.nan, 1, 1], "label": [0] * 6})
    measured = metrics(df, [True, True, False, True, True, False])
    assert measured["false_alarm_episodes"] == 2
    assert measured["false_positive"] == 3
    assert measured["excluded_hours"] == 1
    assert measured["precision"] == 0
    assert measured["recall"] is None
