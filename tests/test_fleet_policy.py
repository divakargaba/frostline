"""Synthetic correctness checks, separate from real-data detection measurements."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.fleet_model import DEFAULT_POLICY
from src.fleet_policy import advance_monitor, iter_observation_contexts, observation_context, operator_assessment


def frame(n=80):
    return pd.DataFrame({"P-TPT": 100., "P-PDG": 150., "P-MON-CKP": 95., "ABER-CKP": 50.},
                        index=pd.date_range("2024-01-01", periods=n, freq="min"))


def result(hydrate=.05, other=.05, *, missing=(), invalid=(), blocked=False):
    return {"scores": {"normal": 1 - hydrate - other, "hydrate": hydrate, "lookalike": other},
            "quality": {"blocked": blocked, "status": "unavailable" if blocked else "good",
                        "missing": list(missing), "invalid": list(invalid), "unchanged": []}}


def tick(state, data, i, prediction=None):
    return advance_monitor(state, prediction or result(), data.iloc[:i + 1], DEFAULT_POLICY)


def test_persistence_recovery_and_human_check_state_are_independent():
    data, state = frame(), {}
    for i in range(3):
        step = tick(state, data, i, result(.8))
        state = step["state"]
    assert step["status"] == "attention"
    assert step["assessment"]["scenario"] == "hydrate_pattern"
    assert {c["id"] for c in step["assessment"]["checks"]} >= {"confirm_recent_operations", "engineering_review"}
    state.update(acknowledged=True, completed=True, note="Everything normal", checks={"engineering_review": "done"})
    step = tick(state, data, 3, result(.8))
    assert step["status"] == "attention"
    for i in range(4, 8):
        step = tick(step["state"], data, i)
        assert step["status"] == "attention"
    step = tick(step["state"], data, 8)
    assert step["status"] == "normal"
    assert step["state"]["alarm"]["recovered"]


def test_missing_telemetry_preserves_existing_attention_but_does_not_create_process_alarm():
    data, state = frame(), {}
    for i in range(3):
        state = tick(state, data, i, result(.8))["state"]
    missing = result(blocked=True, missing=["P-PDG", "P-TPT", "P-MON-CKP"])
    missing["scores"] = {name: None for name in ["normal", "hydrate", "lookalike"]}
    data.iloc[3, :3] = np.nan
    step = tick(state, data, 3, missing)
    assert step["status"] == "attention"
    assert step["assessment"]["scenario"] == "telemetry_missing"
    assert step["recheck_minutes"] == 1
    assert tick({}, data, 3, missing)["status"] == "unavailable"


def test_invalid_or_partial_pressure_prompts_specific_telemetry_checks():
    data = frame()
    invalid = tick({}, data, 20, result(invalid=["P-PDG"]))
    assert invalid["status"] == "watch"
    assert invalid["assessment"]["scenario"] == "telemetry_invalid"
    assert "P-PDG" in invalid["assessment"]["checks"][0]["label"]
    partial = tick({}, data, 20, result(missing=["P-TPT"]))
    assert partial["status"] == "watch"
    assert partial["assessment"]["scenario"] == "telemetry_partial"
    assert tick({}, data, 20, result(missing=["QGL", "T-PDG"]))["status"] == "normal"


def test_pressure_loss_and_divergence_use_observed_numbers_without_claiming_cause():
    data = frame()
    data.loc[data.index[10:21], ["P-TPT", "P-MON-CKP"]] = np.column_stack([np.linspace(100, 94, 11), np.linspace(95, 89, 11)])
    step = tick({}, data, 20)
    assert step["status"] == "watch"
    assert step["assessment"]["scenario"] == "pressure_loss"
    assert "-6.00 bar" in step["assessment"]["evidence"][0]
    assert "confirm_recent_operations" in {c["id"] for c in step["assessment"]["checks"]}
    data["P-TPT"] = 100.
    step = tick({}, data, 20)
    assert step["status"] == "watch"
    assert step["assessment"]["scenario"] == "pressure_divergence"
    assert "6.00 bar" in step["assessment"]["evidence"][0]


def test_invalid_pressure_sentinels_and_missing_endpoint_cannot_fabricate_trend():
    data = frame()
    data.loc[data.index[10:20], "P-TPT"] = -1e40
    context = observation_context(data.iloc[:21])
    assert context["pressures"]["P-TPT"]["samples"] == 1
    assert not tick({}, data, 20)["signals"]["pressure_changing"]
    data.loc[data.index[20], "P-TPT"] = np.nan
    assert context["timestamp"] == data.index[20].isoformat()
    assert observation_context(data.iloc[:21])["pressures"]["P-TPT"]["change_bar"] is None


def test_pressure_watch_needs_six_observations_and_fixed_relative_threshold():
    data = frame()
    data["P-TPT"] = np.linspace(100, 80, len(data))
    assert not tick({}, data, 4)["signals"]["pressure_changing"]
    assert tick({}, data, 20)["signals"]["pressure_changing"]
    high = frame()
    high["P-TPT"] = 1000 - np.arange(len(high)) * .3
    assert not tick({}, high, 20)["signals"]["pressure_changing"]


def test_other_alarm_persists_recovers_and_resets_streak_across_time_gaps():
    data, state = frame(), {}
    for i in range(3):
        step = tick(state, data, i, result(.05, .8))
        state = step["state"]
    assert step["status"] == "attention" and state["other_active"]
    assert step["assessment"]["scenario"] == "restriction_pattern"
    for i in range(3, 8):
        step = tick(step["state"], data, i)
    assert step["status"] == "normal"
    first = tick({}, data, 0, result(.05, .8))
    second = tick(first["state"], data, 1, result(.05, .8))
    skipped = tick(second["state"], data, 3, result(.05, .8))
    assert skipped["state"]["other_run"] == 1 and skipped["status"] == "watch"


def test_agent_escalation_recovers_only_after_five_fresh_normal_minutes():
    data, state = frame(), {"agent_status": "attention"}
    for i in range(4):
        step = tick(state, data, i)
        state = step["state"]
        assert step["status"] == "attention"
    assert tick(state, data, 4)["status"] == "normal"


def test_state_labels_and_future_values_do_not_leak_or_mutate():
    data, state = frame(), {"revision": 0, "alarm": {}, "priorities": [{"status": "watch"}]}
    original = deepcopy(state)
    expected = tick(state, data, 20)
    poisoned = data.copy()
    poisoned["class"] = 108
    poisoned["state"] = 8
    poisoned.iloc[21:, :4] = 10000
    assert tick(state, poisoned, 20) == expected
    assert state == original
    json.dumps(expected, allow_nan=False)


def test_per_well_state_is_independent_and_timestamp_cannot_repeat():
    data, a, b = frame(), {}, {}
    for i in range(3):
        a = tick(a, data, i, result(.8))["state"]
        b = tick(b, data, i)["state"]
    assert a["status"] == "attention" and b["status"] == "normal"
    with pytest.raises(ValueError, match="increase"):
        tick(a, data, 2)


def test_batched_contexts_match_prefixes_and_ignore_invalid_annotated_values():
    data = frame()
    data["P-TPT"] = np.sin(np.arange(len(data))) * 4 + 100
    data.loc[data.index[3:7], "P-PDG"] = np.nan
    data["invalid_P-TPT"] = 0
    data.loc[data.index[25:30], "invalid_P-TPT"] = 1
    data["class"] = 108
    for i, context in enumerate(iter_observation_contexts(data)):
        assert context == observation_context(data.iloc[:i + 1])


def test_assessment_refresh_does_not_advance_or_change_state():
    data = frame()
    step = tick({}, data, 10, result(.8))
    before = deepcopy(step["state"])
    output = operator_assessment(result(.8), data.iloc[:11], step["status"], step["state"], DEFAULT_POLICY)
    assert output == step["assessment"]
    assert step["state"] == before
