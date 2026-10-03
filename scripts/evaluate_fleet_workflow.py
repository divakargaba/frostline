"""Measure the exact non-LLM monitor on all recordings from four excluded wells.

No training or policy selection occurs here. Annotation arrays are passed only
to scoring after the sensor-only monitor has produced its chronological output.
Synthetic lifecycle correctness checks are reported separately from real data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fleet_model import (
    BUNDLE_PATH, CLASS_NAMES, EXCLUDED_WELLS, PRESSURES, RAW, SENSORS,
    _sanitize, feature_frame, infer_window, load_bundle, prepare_recording,
)
from src.fleet_policy import POLICY_VERSION, advance_monitor, iter_observation_contexts

REPORT_PATH = ROOT / "data/processed/research/fleet_workflow_report.json"
EVENT_LABELS = {"hydrate": (8, 108), "restriction": (6, 106), "scaling": (7, 107)}


def _number(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def batch_inference_results(observed_frame, bundle):
    """Batched features/scores, causal per-minute evidence identical to runtime.

    The caller supplies a complete minute grid. Features use trailing windows;
    no backwards fill, labels or whole-record sensor decisions are introduced.
    Quality uses the same 60-minute window as infer_window. A sampled streaming
    parity check is mandatory in the CLI before these scores are reported.
    """
    sensor, runtime_invalid = _sanitize(observed_frame.reindex(columns=SENSORS).astype(float), converted=True)
    features = feature_frame(sensor, bundle).reindex(columns=bundle["feature_names"])
    probabilities = bundle["model"].predict_proba(features)
    values = sensor.to_numpy()
    missing = ~np.isfinite(values)
    invalid = runtime_invalid.to_numpy().copy()
    for j, name in enumerate(SENSORS):
        if f"invalid_{name}" in observed_frame:
            invalid[:, j] |= observed_frame[f"invalid_{name}"].fillna(0).to_numpy() > 0
    # pandas rolling min/max/count is causal and avoids a Python nunique per row.
    rolling = sensor.rolling("60min", min_periods=1)
    window_len = pd.Series(1, index=sensor.index).rolling("60min", min_periods=1).sum().to_numpy()
    unchanged = (rolling.max().to_numpy() == rolling.min().to_numpy()) & (rolling.count().to_numpy() >= 54) & (window_len[:, None] >= 60)
    pressure_indices = [SENSORS.index(name) for name in PRESSURES]
    tpt_index, mon_index = SENSORS.index("P-TPT"), SENSORS.index("P-MON-CKP")
    available = ~missing[:, pressure_indices]
    blocked = ~available.any(axis=1)
    line_available = ~missing[:, tpt_index] & ~missing[:, mon_index]
    stamps = sensor.index
    # Match ``index > current - 60 minutes`` exactly, including any source gaps.
    starts = stamps.searchsorted(stamps - pd.Timedelta(minutes=60), side="right")
    trend = {}
    for name in ["P-TPT", "P-MON-CKP", "T-TPT", "QGL"]:
        column = values[:, SENSORS.index(name)]
        positions = np.flatnonzero(np.isfinite(column))
        first = np.searchsorted(positions, starts)
        last = np.searchsorted(positions, np.arange(len(sensor)), side="right") - 1
        valid = (last - first >= 1) & (first < len(positions))
        changes = np.full(len(sensor), np.nan)
        changes[valid] = column[positions[last[valid]]] - column[positions[first[valid]]]
        trend[name] = changes
    for i, stamp in enumerate(stamps):
        invalid_names = [name for j, name in enumerate(SENSORS) if invalid[i, j]]
        available_names = [name for j, name in enumerate(PRESSURES) if available[i, j]]
        status = "unavailable" if blocked[i] else "degraded" if invalid_names or not line_available[i] else "good"
        yield {
            "scores": {name: None if blocked[i] else float(probabilities[i, j]) for j, name in enumerate(CLASS_NAMES)},
            "quality": {"status": status, "blocked": bool(blocked[i]), "available_pressure": available_names,
                        "line_pressure_available": bool(line_available[i]),
                        "missing": [name for j, name in enumerate(SENSORS) if missing[i, j]],
                        "invalid": invalid_names,
                        "unchanged": [name for j, name in enumerate(SENSORS) if name.startswith(("P-", "T-")) and unchanged[i, j]],
                        "summary": "No usable pressure evidence." if blocked[i] else f"{len(available_names)}/3 pressure channels available; {len(invalid_names)} invalid channels. Unchanged values alone are not a fault."},
            "evidence": {"timestamp": stamp.isoformat(), "model_id": bundle["model_id"],
                         "sensor_values": {name: _number(values[i, j]) for j, name in enumerate(SENSORS)},
                         "line_pressure_difference_bar": _number(values[i, tpt_index] - values[i, mon_index]),
                         "trailing_change_60min": {name: _number(changes[i]) for name, changes in trend.items()},
                         "history_minutes": min(i + 1, 180),
                         "score_note": "Uncalibrated model scores; QGL is gas-lift injection, not oil production."},
        }


def verify_streaming_parity(sensor_frame, bundle, results, sample_indices=None):
    """Fail rather than publish a report if batched predictions differ from live."""
    if sample_indices is None:
        sample_indices = sorted({0, min(10, len(sensor_frame) - 1), min(59, len(sensor_frame) - 1),
                                 min(60, len(sensor_frame) - 1), min(180, len(sensor_frame) - 1),
                                 len(sensor_frame) // 2, len(sensor_frame) - 1})
    for i in sample_indices:
        actual = infer_window(sensor_frame.iloc[max(0, i - 179):i + 1], bundle)
        expected = results[i]
        if actual["quality"] != expected["quality"]:
            raise AssertionError(f"Quality batch/stream mismatch at minute {i}")
        if actual["evidence"] != expected["evidence"]:
            raise AssertionError(f"Evidence batch/stream mismatch at minute {i}")
        for name in CLASS_NAMES:
            a, b = actual["scores"][name], expected["scores"][name]
            if a is None or b is None:
                if a != b:
                    raise AssertionError(f"Missing score mismatch at minute {i}")
            elif not np.isclose(a, b, rtol=1e-9, atol=1e-9):
                raise AssertionError(f"Score batch/stream mismatch at minute {i}: {name}")
    return len(sample_indices)


def _episodes(mask):
    values = np.asarray(mask, dtype=bool)
    return int((values & ~np.r_[False, values[:-1]]).sum())


def _score_outputs(labels, index, statuses):
    """Evaluator-only use of annotations; status decisions already exist."""
    labels = np.asarray(labels)
    statuses = np.asarray(statuses)
    normal = labels == 0
    known = np.isin(labels, [0, 6, 106, 7, 107, 8, 108])
    attention, watch, unavailable = statuses == "attention", statuses == "watch", statuses == "unavailable"
    flag = attention | watch
    metrics = {"observed_minutes": len(labels), "labelled_minutes": int(known.sum()), "normal_minutes": int(normal.sum()),
               "attention_minutes": int(attention.sum()), "watch_minutes": int(watch.sum()), "unavailable_minutes": int(unavailable.sum()),
               "false_attention_minutes": int((attention & normal).sum()), "false_watch_minutes": int((watch & normal).sum()),
               "false_attention_episodes": _episodes(attention & normal), "false_watch_episodes": _episodes(watch & normal),
               "review_episodes": _episodes(statuses != "normal"), "process_flag_episodes": _episodes(flag),
               "normal_review_episodes": _episodes((statuses != "normal") & normal), "events": {}}
    for name, classes in EVENT_LABELS.items():
        truth = np.isin(labels, classes)
        locations = np.flatnonzero(truth)
        overlap = np.flatnonzero(truth & flag)
        attention_overlap = np.flatnonzero(truth & attention)
        onset = int(locations[0]) if len(locations) else None
        first = int(overlap[0]) if len(overlap) else None
        new_flags = truth & flag & ~np.r_[False, flag[:-1]]
        new_locations = np.flatnonzero(new_flags)
        delay = (index[first] - index[onset]).total_seconds() / 60 if first is not None else None
        new_delay = (index[new_locations[0]] - index[onset]).total_seconds() / 60 if len(new_locations) else None
        metrics["events"][name] = {"recording_present": bool(len(locations)), "flagged": bool(len(overlap)),
                                    "attention": bool(len(attention_overlap)), "event_minutes": int(truth.sum()),
                                    "flagged_minutes": int((truth & flag).sum()), "attention_minutes": int((truth & attention).sum()),
                                    "preexisting_at_onset": bool(onset is not None and onset > 0 and flag[onset - 1] and flag[onset]),
                                    "first_flag_delay_minutes": delay, "new_flag_delay_minutes": new_delay,
                                    "new_flag_in_event": bool(len(new_locations))}
    return metrics


def evaluate_recording(frame, bundle):
    """One frozen-model run; labels are removed before every monitoring call."""
    sensor = frame.reindex(columns=[*SENSORS, *[f"invalid_{name}" for name in SENSORS]])
    results = list(batch_inference_results(sensor, bundle))
    checked = verify_streaming_parity(sensor, bundle, results)
    state, statuses, model_only, scenarios = {}, [], [], Counter()
    seen_checks, checks_created, previous = set(), 0, "normal"
    review_assessments, due = 0, 0
    for i, (result, context) in enumerate(zip(results, iter_observation_contexts(sensor))):
        step = advance_monitor(state, result, context, bundle["policy"])
        state = step["state"]
        status = step["status"]
        statuses.append(status)
        model_only.append("attention" if state["alarm"]["alarm"] else "unavailable" if result["quality"]["blocked"] else "normal")
        scenarios[step["assessment"]["scenario"]] += 1
        if status != "normal":
            if previous == "normal":
                seen_checks = set()
            new_checks = {c["id"] for c in step["assessment"]["checks"]} - seen_checks
            checks_created += len(new_checks)
            seen_checks.update(new_checks)
            if step["material"] or i >= due:
                review_assessments += 1
                due = i + step["recheck_minutes"]
        else:
            due = max(due, i + step["recheck_minutes"])
        previous = status
    labels = frame["class"].to_numpy()
    workflow = _score_outputs(labels, frame.index, statuses)
    workflow.update(check_items_created=checks_created, immediate_review_assessments=review_assessments)
    return {"workflow": workflow, "model_only": _score_outputs(labels, frame.index, model_only),
            "assessment_minutes_by_scenario": dict(scenarios), "streaming_parity_samples": checked}


def aggregate(records, key):
    metrics = [item[key] for item in records]
    numeric_keys = [k for k, value in metrics[0].items() if isinstance(value, (int, float))]
    output = {k: sum(item[k] for item in metrics) for k in numeric_keys}
    output["events"] = {}
    for name in EVENT_LABELS:
        events = [m["events"][name] for m in metrics]
        delays = [v["first_flag_delay_minutes"] for v in events if v["first_flag_delay_minutes"] is not None]
        new_delays = [v["new_flag_delay_minutes"] for v in events if v["new_flag_delay_minutes"] is not None]
        output["events"][name] = {
            "recordings": sum(v["recording_present"] for v in events), "recordings_flagged": sum(v["flagged"] for v in events),
            "recordings_attention": sum(v["attention"] for v in events),
            "preexisting_at_onset": sum(v["preexisting_at_onset"] for v in events),
            "recordings_with_new_flag": sum(v["new_flag_in_event"] for v in events),
            "event_minutes": sum(v["event_minutes"] for v in events), "flagged_minutes": sum(v["flagged_minutes"] for v in events),
            "mean_first_flag_delay_minutes": float(np.mean(delays)) if delays else None,
            "mean_new_flag_delay_minutes": float(np.mean(new_delays)) if new_delays else None,
        }
    days = output["normal_minutes"] / 1440
    output["false_attention_minutes_per_normal_day"] = output["false_attention_minutes"] / days if days else None
    output["false_watch_minutes_per_normal_day"] = output["false_watch_minutes"] / days if days else None
    return output


def synthetic_lifecycle_checks():
    """Actual assertions against the shared policy, not measured field outcomes."""
    from src.fleet_model import DEFAULT_POLICY
    clock = pd.date_range("2024-01-01", periods=40, freq="min")
    data = pd.DataFrame({"P-TPT": 100., "P-PDG": 150., "P-MON-CKP": 95., "ABER-CKP": 50.}, index=clock)

    def result(hydrate=.05, other=.05, blocked=False):
        return {"scores": {"normal": None if blocked else 1-hydrate-other, "hydrate": None if blocked else hydrate, "lookalike": None if blocked else other},
                "quality": {"blocked": blocked, "status": "unavailable" if blocked else "good", "invalid": [], "missing": list(PRESSURES) if blocked else []}}

    def advance(state, i, prediction=None, observed=None):
        return advance_monitor(state, prediction or result(), (data if observed is None else observed).iloc[:i+1], DEFAULT_POLICY)

    def persistence():
        a, b = {}, {}
        for i in range(3):
            a = advance(a, i, result(.8))["state"]
            b = advance(b, i)["state"]
        assert a["status"] == "attention" and b["status"] == "normal"

    def outage():
        state = {}
        for i in range(3):
            state = advance(state, i, result(.8))["state"]
        missing = advance(state, 3, result(blocked=True))
        assert missing["status"] == "attention" and missing["assessment"]["scenario"] == "telemetry_missing"
        assert advance({}, 3, result(blocked=True))["status"] == "unavailable"

    def recovery():
        state = {}
        for i in range(3):
            state = advance(state, i, result(.8))["state"]
        for i in range(3, 7):
            state = advance(state, i)["state"]
            assert state["status"] == "attention"
        assert advance(state, 7)["status"] == "normal"

    def checks_not_recovery():
        state = {}
        for i in range(3):
            state = advance(state, i, result(.8))["state"]
        state.update(acknowledged=True, completed=True, operator_observation="normal, clear alarm", checks={"engineering_review": "done"})
        assert advance(state, 3, result(.8))["status"] == "attention"

    def labels_cannot_decide():
        poisoned = data.copy()
        poisoned["class"] = 108
        poisoned["state"] = 8
        poisoned.iloc[10:, :4] = 10000
        assert advance({}, 9, observed=poisoned) == advance({}, 9)

    def targeted_pressure_checks():
        changed = data.copy()
        changed.loc[clock[:11], "P-MON-CKP"] = np.linspace(95, 87, 11)
        step = advance({}, 10, observed=changed)
        assert step["status"] == "watch" and step["assessment"]["scenario"] == "pressure_divergence"
        assert step["recheck_minutes"] == 1
        assert "confirm_recent_operations" in {c["id"] for c in step["assessment"]["checks"]}

    cases = [
        ("independent_persistence", "Independent wells and persistent detection", persistence, "One synthetic well reaches attention after three high-score minutes while the other remains normal."),
        ("missing_telemetry", "Missing readings preserve unresolved concern", outage, "A data outage preserves existing attention; an outage alone becomes telemetry unavailable, not a process diagnosis."),
        ("observed_recovery", "Recovery requires fresh normal readings", recovery, "Four low-score minutes retain the incident; the fifth clears the numerical concern."),
        ("checks_not_recovery", "Human review does not clear physical concern", checks_not_recovery, "Acknowledgment, completed checks and an operator note cannot override persistent measured risk."),
        ("causal_boundary", "Labels and future rows cannot influence a check", labels_cannot_decide, "Changing source labels and unseen future readings leaves the current assessment identical."),
        ("targeted_checks", "Pressure divergence produces targeted follow-up", targeted_pressure_checks, "A measured pressure divergence creates comparison and recent-operation checks with a one-minute recheck."),
    ]
    outcomes = []
    for identifier, name, run, evidence in cases:
        try:
            run()
            outcomes.append({"id": identifier, "name": name, "status": "pass", "evidence": evidence, "synthetic": True})
        except AssertionError:
            outcomes.append({"id": identifier, "name": name, "status": "fail", "evidence": "Synthetic lifecycle assertion failed; inspect the shared monitoring policy.", "synthetic": True})
    return outcomes


def run(output=REPORT_PATH):
    bundle = load_bundle()
    excluded = set(EXCLUDED_WELLS)
    if excluded & set(bundle["training_wells"]) or not excluded.issubset(bundle["excluded_wells"]):
        raise ValueError("All four evaluation wells must remain excluded from training")
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    recordings = []
    selected = [item for item in manifest["files"] if item["well"] in excluded]
    if len(selected) != 11:
        raise ValueError("This frozen four-well evaluation expects the approved 11-recording subset")
    for item in selected:
        path = RAW / str(item["class"]) / item["name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path.name}")
        frame = prepare_recording(path)
        scored = evaluate_recording(frame, bundle)
        recordings.append({"file": item["name"], "well_id": item["well"], **scored})
        print(f"Evaluated {path.name}: {len(frame)} observed minutes", flush=True)
    workflow, baseline = aggregate(recordings, "workflow"), aggregate(recordings, "model_only")
    scenarios = synthetic_lifecycle_checks()
    metrics = {"recordings": len(recordings), "observed_minutes": workflow["observed_minutes"], "labelled_minutes": workflow["labelled_minutes"],
               "normal_minutes": workflow["normal_minutes"], "workflow_attention_minutes": workflow["attention_minutes"],
               "workflow_watch_minutes": workflow["watch_minutes"], "workflow_unavailable_minutes": workflow["unavailable_minutes"],
               "workflow_false_attention_minutes": workflow["false_attention_minutes"], "workflow_false_watch_minutes": workflow["false_watch_minutes"],
               "workflow_review_episodes": workflow["review_episodes"], "workflow_check_items_created": workflow["check_items_created"],
               "model_only_false_attention_minutes": baseline["false_attention_minutes"]}
    for name in EVENT_LABELS:
        metrics[f"{name}_events_detected"] = workflow["events"][name]["recordings_flagged"]
        metrics[f"{name}_events_total"] = workflow["events"][name]["recordings"]
    report = {
        "status": "complete", "generated_at": datetime.now(timezone.utc).isoformat(), "model_id": bundle["model_id"],
        "policy_version": POLICY_VERSION, "policy": bundle["policy"],
        "provenance": {"dataset_revision": manifest["revision"], "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                       "bundle_sha256": hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest(),
                       "monitor_sha256": hashlib.sha256((ROOT / "src/fleet_policy.py").read_bytes()).hexdigest(),
                       "excluded_wells": sorted(excluded), "streaming_parity_samples": sum(r["streaming_parity_samples"] for r in recordings)},
        "summary": {"passed": sum(s["status"] == "pass" for s in scenarios), "total": len(scenarios)},
        "metrics": metrics, "comparisons": {"model_only": baseline, "workflow": workflow}, "scenarios": scenarios, "recordings": recordings,
        "definitions": {
            "coverage": "A labelled recording is flagged when watch or attention overlaps at least one labelled event minute; this is not correct causal diagnosis.",
            "delay": "Time from the first labelled event minute to the first overlapping flag; pre-existing flags are separately counted and are not new event discoveries. New-flag delay excludes already-open flags.",
            "false_burden": "Attention and watch minutes on label 0 are reported separately. Telemetry unavailability is not counted as a process alarm.",
            "work_counts": "Review episodes are transitions into a non-normal condition. Check items count distinct requested check IDs within each continuous incident, not human actions or time saved.",
            "reassessments": "Immediate deterministic review opportunities assume zero processing delay; these are not LLM requests or a claim about actual scheduler/network timing.",
            "baseline": "Same frozen hydrate model and persistence policy, without other-class, relative-pressure or partial-telemetry review triggers.",
            "lifecycle": "Synthetic policy assertions are correctness evidence only; they do not contribute any real-data score.",
        },
        "limits": [
            "Internal evaluation of the previously explored 31-recording subset; all 11 recordings of four demo wells are excluded from fitting and policy selection. This is not untouched external validation.",
            "Workflow thresholds were fixed before this report. No threshold, model or policy was selected using these held-out results.",
            "Watch or attention overlap measures screening coverage, not correct hydrate, restriction or scaling diagnosis; all hydrate evaluation recordings belong to one well.",
            "The full shared numerical watcher and deterministic assessments are evaluated. LLM decisions, operator response, network latency and actual dispatch are not scored.",
            "New relative-pressure and partial-telemetry checks can increase watch burden. Acknowledgment and completed checks do not suppress measured concerns.",
            "Review/check counts do not establish reduced operator effort or time savings; an operator study is still required.",
            "Full recordings start without an assumed healthy warm-up. The dashboard uses shorter fixed excerpts with preceding observed history, so the score is not a count from one three-hour presentation run.",
            "Guidance is a project review aid, not an approved site procedure, thermodynamic diagnosis or equipment-control instruction.",
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(output), "model_id": report["model_id"], "summary": report["summary"], "metrics": metrics}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    run(parser.parse_args().output)
