"""A frozen, reproducible 3W model for four previously excluded demo wells.

Raw Parquets stay unchanged. Labels are used only by training/evaluation;
infer_window selects an explicit sensor allowlist and only the supplied prefix.
This remains a bounded, previously explored subset, not external validation.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from src.load import SENSORS
from src.real_pilot import causal_features, minute_frame, targets

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/research"
BUNDLE_PATH = ROOT / "models/fleet_model.pkl"
REPORT_PATH = ROOT / "data/processed/research/fleet_report.json"
SCHEMA_VERSION = "fleet-causal-minute-v1"
PRESSURES = ["P-PDG", "P-TPT", "P-MON-CKP"]
CLASS_NAMES = ["normal", "hydrate", "lookalike"]
DEMO_RECORDINGS = [
    {"well_id": "WELL-00001", "recording_id": "WELL-00001_20170201010207", "source_class": 0, "display_name": "Well 01", "path": str(RAW / "0/WELL-00001_20170201010207.parquet")},
    {"well_id": "WELL-00002", "recording_id": "WELL-00002_20140212160333", "source_class": 6, "display_name": "Well 02", "path": str(RAW / "6/WELL-00002_20140212160333.parquet")},
    {"well_id": "WELL-00006", "recording_id": "WELL-00006_20180618103000", "source_class": 7, "display_name": "Well 06", "path": str(RAW / "7/WELL-00006_20180618103000.parquet")},
    {"well_id": "WELL-00019", "recording_id": "WELL-00019_20120601165020", "source_class": 8, "display_name": "Well 19", "path": str(RAW / "8/WELL-00019_20120601165020.parquet")},
]
EXCLUDED_WELLS = sorted(item["well_id"] for item in DEMO_RECORDINGS)
DEFAULT_POLICY = {"activation_threshold": .5, "persistence_minutes": 3,
                  "recovery_minutes": 5, "recovery_threshold": .4}


def _sanitize(sensor, converted=False):
    invalid = sensor.notna() & ~np.isfinite(sensor)
    for name in SENSORS:
        magnitude_limit = 1e25 if converted and name.startswith("P-") else 1e30
        invalid[name] |= sensor[name].abs().ge(magnitude_limit)
        if name.startswith("T-"):
            invalid[name] |= sensor[name].lt(-273.15)
        if name.startswith("ABER-"):
            invalid[name] |= sensor[name].notna() & ~sensor[name].between(0, 100)
    return sensor.mask(invalid), invalid


def prepare_recording(path) -> pd.DataFrame:
    """Sanitize a derived view, retaining the full minute grid and annotations.

    Minute bins are labelled when fully available. No whole-file quality
    decision, future interpolation, label-based dropping, or gap filling.
    """
    raw = pd.read_parquet(path)
    if "timestamp" in raw:
        raw = raw.set_index("timestamp")
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    if raw.index.has_duplicates:
        raise ValueError("Duplicate source timestamps require an explicit resolution")
    sensor = raw.reindex(columns=SENSORS).astype(float)
    sensor, invalid = _sanitize(sensor)
    for name in SENSORS:
        if name.startswith("P-"):
            sensor[name] /= 1e5
    frame = sensor.resample("1min", label="right", closed="left").mean()
    invalid_minute = invalid.astype(float).resample("1min", label="right", closed="left").mean()
    for name in SENSORS:
        frame[f"invalid_{name}"] = invalid_minute[name]
    if "class" in raw:
        frame["class"] = raw["class"].resample("1min", label="right", closed="left").agg(
            lambda values: values.mode().iloc[0] if values.notna().any() else np.nan)
    else:
        frame["class"] = np.nan
    return frame


def _validate_prefix(frame):
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("A nonempty timestamp-indexed sensor prefix is required")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Sensor timestamps must be unique and increasing")


def _number(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def infer_window(observed_frame, bundle=None) -> dict:
    """Infer at the last supplied timestamp. Never inspect labels or a future row."""
    _validate_prefix(observed_frame)
    bundle = load_bundle() if bundle is None else bundle
    sensors, runtime_invalid = _sanitize(observed_frame.reindex(columns=SENSORS).astype(float), converted=True)
    # The caller can supply only already-prepared sensor columns. Labels ignored.
    current = sensors.iloc[-1]
    available = [name for name in PRESSURES if pd.notna(current[name])]
    missing = [name for name in SENSORS if pd.isna(current[name])]
    invalid = [name for name in SENSORS if runtime_invalid[name].iloc[-1]
               or (f"invalid_{name}" in observed_frame and observed_frame[f"invalid_{name}"].iloc[-1] > 0)]
    recent = sensors.loc[sensors.index > sensors.index[-1] - pd.Timedelta(minutes=60)]
    unchanged = [name for name in SENSORS if name.startswith(("P-", "T-"))
                 and len(recent) >= 60 and recent[name].notna().sum() >= 54
                 and recent[name].nunique(dropna=True) == 1]
    blocked = not available
    line_available = pd.notna(current["P-TPT"]) and pd.notna(current["P-MON-CKP"])
    status = "unavailable" if blocked else "degraded" if invalid or not line_available else "good"
    features = causal_features(recent).reindex(columns=bundle["feature_names"])
    if blocked:
        scores = {name: None for name in CLASS_NAMES}
    else:
        probability = bundle["model"].predict_proba(features.iloc[[-1]])[0]
        scores = {name: float(probability[index]) for index, name in enumerate(CLASS_NAMES)}
    trend = {}
    for name in ["P-TPT", "P-MON-CKP", "T-TPT", "QGL"]:
        valid = recent[name].dropna()
        trend[name] = _number(valid.iloc[-1] - valid.iloc[0]) if len(valid) > 1 else None
    return {
        "scores": scores,
        "quality": {"status": status, "blocked": blocked, "available_pressure": available,
                    "line_pressure_available": bool(line_available),
                    "missing": missing, "invalid": invalid, "unchanged": unchanged,
                    "summary": "No usable pressure evidence." if blocked else f"{len(available)}/3 pressure channels available; {len(invalid)} invalid channels. Unchanged values alone are not a fault."},
        "evidence": {"timestamp": sensors.index[-1].isoformat(), "model_id": bundle["model_id"],
                     "sensor_values": {name: _number(current[name]) for name in SENSORS},
                     "line_pressure_difference_bar": _number(current["P-TPT"] - current["P-MON-CKP"]),
                     "trailing_change_60min": trend, "history_minutes": len(sensors),
                     "score_note": "Uncalibrated model scores; QGL is gas-lift injection, not oil production."},
    }


def advance_alarm(state, probability, pressure_available, timestamp, policy=None):
    """Shared causal alarm state: persistence, recovery hysteresis, and data gaps.

    Missing evidence suspends the alarm signal but preserves an already-open
    incident. Recovery requires five new, consecutive, observed low-score minutes.
    """
    policy = DEFAULT_POLICY if policy is None else policy
    state = dict(state or {})
    now = pd.Timestamp(timestamp)
    previous = state.get("last_t")
    if previous is not None and now <= pd.Timestamp(previous):
        raise ValueError("Alarm timestamps must increase")
    if previous is not None and now - pd.Timestamp(previous) != pd.Timedelta(minutes=1):
        state.update(activation_streak=0, recovery_streak=0)
    state.setdefault("active", False)
    was_active = bool(state["active"])
    state.update(triggered=False, recovered=False)
    state.setdefault("activation_streak", 0)
    state.setdefault("recovery_streak", 0)
    state["last_t"] = now.isoformat()
    valid = bool(pressure_available and probability is not None and np.isfinite(probability))
    state["suspended"] = not valid
    if not valid:
        state.update(activation_streak=0, recovery_streak=0, alarm=False)
        return state
    if state["active"]:
        state["recovery_streak"] = state["recovery_streak"] + 1 if probability < policy["recovery_threshold"] else 0
        if state["recovery_streak"] >= policy["recovery_minutes"]:
            state.update(active=False, activation_streak=0, recovery_streak=0)
    else:
        state["activation_streak"] = state["activation_streak"] + 1 if probability >= policy["activation_threshold"] else 0
        if state["activation_streak"] >= policy["persistence_minutes"]:
            state.update(active=True, recovery_streak=0)
    state["alarm"] = bool(state["active"])
    state["triggered"] = bool(state["active"] and not was_active)
    state["recovered"] = bool(was_active and not state["active"])
    return state


def alarm_series(frame, probabilities, policy=None):
    available = frame.reindex(columns=PRESSURES).notna().any(axis=1).to_numpy()
    state, alarms = {}, []
    for stamp, probability, valid in zip(frame.index, probabilities, available):
        state = advance_alarm(state, probability, valid, stamp, policy)
        alarms.append(state["alarm"])
    return np.asarray(alarms, dtype=bool)


def recording_metrics(frame, probabilities, policy=None):
    y = targets(frame)
    known = y.notna().to_numpy()
    truth = y.eq(1).to_numpy()
    normal = y.eq(0).to_numpy()
    other = y.eq(2).to_numpy()
    alarm = alarm_series(frame, probabilities[:, 1], policy)
    false = alarm & normal
    starts = alarm & ~np.r_[False, alarm[:-1]]
    positive_indices = np.flatnonzero(truth)
    first_positive = int(positive_indices[0]) if len(positive_indices) else None
    preexisting = bool(first_positive is not None and first_positive > 0 and alarm[first_positive - 1] and alarm[first_positive])
    new_event_alerts = frame.index[starts & truth]
    established = frame.index[frame["class"].eq(8)]
    lead = (established[0] - new_event_alerts[0]).total_seconds() / 60 if len(established) and len(new_event_alerts) else None
    delay = (new_event_alerts[0] - frame.index[first_positive]).total_seconds() / 60 if first_positive is not None and len(new_event_alerts) else None
    return {"labelled_minutes": int(known.sum()), "normal_minutes": int(normal.sum()),
            "hydrate_minutes": int(truth.sum()), "missed_hydrate_minutes": int((truth & ~alarm).sum()),
            "false_alarm_minutes": int(false.sum()),
            "false_alarm_episodes": int((false & ~np.r_[False, false[:-1]]).sum()),
            "hydrate_event": bool(truth.any()), "detected": bool((alarm & truth).any()),
            "new_event_alarm": bool(len(new_event_alerts)), "preexisting_alarm_at_onset": preexisting,
            "lead_minutes": lead, "detection_delay_minutes": delay, "early": bool(lead is not None and lead > 0),
            "lookalike_recording": bool(other.any()), "lookalike_flagged": bool((alarm & other).any()),
            "classification_confusion": confusion_matrix(y[known], probabilities[known].argmax(axis=1), labels=[0, 1, 2]).tolist()}


def aggregate_metrics(records):
    total = {key: sum(r[key] for r in records) for key in ["labelled_minutes", "normal_minutes", "hydrate_minutes", "missed_hydrate_minutes", "false_alarm_minutes", "false_alarm_episodes"]}
    for target, key in [("hydrate_events", "hydrate_event"), ("events_detected", "detected"), ("new_event_alarms", "new_event_alarm"), ("preexisting_alarms_at_onset", "preexisting_alarm_at_onset"), ("early_events", "early"), ("lookalike_recordings", "lookalike_recording"), ("lookalikes_flagged", "lookalike_flagged")]:
        total[target] = sum(bool(r[key]) for r in records)
    total["event_recall"] = total["events_detected"] / total["hydrate_events"] if total["hydrate_events"] else None
    total["alarm_minute_recall"] = 1 - total["missed_hydrate_minutes"] / total["hydrate_minutes"] if total["hydrate_minutes"] else None
    normal_days = total["normal_minutes"] / 1440
    total["false_alarms_per_normal_day"] = total["false_alarm_episodes"] / normal_days if normal_days else None
    total["false_alarm_minutes_per_normal_day"] = total["false_alarm_minutes"] / normal_days if normal_days else None
    total["illustrative_cost"] = 5 * total["missed_hydrate_minutes"] + total["false_alarm_minutes"]
    delays = [r["detection_delay_minutes"] for r in records if r["detection_delay_minutes"] is not None]
    total["mean_detection_delay_minutes"] = float(np.mean(delays)) if delays else None
    total["classification_confusion"] = np.asarray([r["classification_confusion"] for r in records]).sum(axis=0).tolist()
    return total


def _model():
    return LGBMClassifier(n_estimators=80, num_leaves=15, learning_rate=.05,
                          min_child_samples=30, class_weight="balanced", random_state=42,
                          n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)


def _fit(frames):
    x = pd.concat([causal_features(frame) for frame in frames], ignore_index=True)
    y = pd.concat([targets(frame) for frame in frames], ignore_index=True)
    known = y.notna()
    if set(y[known]) != {0, 1, 2}:
        raise ValueError("Training partition must contain all three target classes")
    model = _model().fit(x[known], y[known].astype(int))
    return model, list(x.columns)


def _predict(model, frame):
    return model.predict_proba(causal_features(frame))


def _score_frames(frames, probabilities, policy):
    return aggregate_metrics([recording_metrics(frame, p, policy) for frame, p in zip(frames, probabilities)])


def candidate_qualifies(candidate, incumbent):
    """Require a Pareto improvement in event detection and false-alarm duration."""
    return (candidate["events_detected"] >= incumbent["events_detected"]
            and candidate["false_alarm_minutes"] <= incumbent["false_alarm_minutes"]
            and (candidate["events_detected"] > incumbent["events_detected"]
                 or candidate["false_alarm_minutes"] < incumbent["false_alarm_minutes"]))


def _candidate_key(score, policy):
    delay = score["mean_detection_delay_minutes"]
    return (-score["events_detected"], score["false_alarm_minutes"], score["false_alarm_episodes"],
            delay if delay is not None else float("inf"), policy["persistence_minutes"], policy["activation_threshold"])


def train_bundle() -> dict:
    """Train only on the other 17 wells; tune on their grouped out-of-fold data."""
    manifest_path = RAW / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    items, frames = [], []
    for item in manifest["files"]:
        path = RAW / str(item["class"]) / item["name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {path.name}")
        items.append(item)
        frames.append(prepare_recording(path))
    train_ids = [i for i, item in enumerate(items) if item["well"] not in EXCLUDED_WELLS]
    test_ids = [i for i, item in enumerate(items) if item["well"] in EXCLUDED_WELLS]
    training_wells = sorted({items[i]["well"] for i in train_ids})
    if set(training_wells) & set(EXCLUDED_WELLS):
        raise AssertionError("Demo well leaked into training")
    groups = np.asarray([items[i]["well"] for i in train_ids])
    train_frames = [frames[i] for i in train_ids]
    oof = [None] * len(train_ids)
    audit = []
    for fold, (a, b) in enumerate(GroupKFold(3).split(train_ids, groups=groups), 1):
        print(f"Fitting grouped validation fold {fold}/3", flush=True)
        model, feature_names = _fit([train_frames[j] for j in a])
        for j in b:
            oof[j] = _predict(model, train_frames[j])
        audit.append({"fold": fold, "train_wells": sorted(set(groups[a])), "validation_wells": sorted(set(groups[b]))})
    baseline_validation = _score_frames(train_frames, oof, DEFAULT_POLICY)
    winner, winner_score, trials = dict(DEFAULT_POLICY), baseline_validation, []
    for threshold in [.5, .6, .7, .8, .9]:
        for persistence in [3, 5, 10]:
            policy = {"activation_threshold": threshold, "persistence_minutes": persistence,
                      "recovery_threshold": round(threshold - .1, 2), "recovery_minutes": 5}
            score = _score_frames(train_frames, oof, policy)
            qualifies = candidate_qualifies(score, baseline_validation)
            improve = qualifies and _candidate_key(score, policy) < _candidate_key(winner_score, winner)
            if improve:
                winner, winner_score = policy, score
            trials.append({"policy": policy, "metrics": score, "qualifies": qualifies, "accepted": improve,
                           "reason": "No more missed events or false-alarm minutes; strict improvement over the validation incumbent. Ties use episodes, delay, then shorter persistence." if improve else "Retain current choice: no qualifying Pareto improvement, or an already better candidate exists."})
    print("Fitting frozen model and baseline on the 17 eligible wells", flush=True)
    model, feature_names = _fit(train_frames)
    raw_frames = [minute_frame(RAW / str(items[i]["class"]) / items[i]["name"]) for i in range(len(items))]
    baseline_model, _ = _fit([raw_frames[i] for i in train_ids])
    evaluation_frames = [frames[i] for i in test_ids]
    quality_p = [_predict(model, frame) for frame in evaluation_frames]
    raw_p = [_predict(baseline_model, raw_frames[i]) for i in test_ids]
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    identity = json.dumps({"schema": SCHEMA_VERSION, "manifest": manifest_hash, "training_wells": training_wells, "policy": winner}, sort_keys=True)
    model_id = "fleet-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
    bundle = {"model": model, "model_id": model_id, "schema_version": SCHEMA_VERSION,
              "feature_names": feature_names, "sensors": list(SENSORS), "class_names": CLASS_NAMES,
              "policy": winner, "policy_version": model_id, "training_wells": training_wells,
              "excluded_wells": EXCLUDED_WELLS, "manifest_sha256": manifest_hash,
              "source_revision": manifest["revision"], "license": manifest["license"],
              "training_recordings": [items[i]["name"] for i in train_ids]}
    report = {"model_id": model_id, "schema_version": SCHEMA_VERSION, "manifest_sha256": manifest_hash,
              "protocol": "Fixed four demo wells excluded completely; 3-fold grouped validation on the other 17 wells; fixed LightGBM; 15 threshold/persistence candidates; five-minute recovery hysteresis.",
              "training_wells": training_wells, "excluded_wells": EXCLUDED_WELLS,
              "training_recordings": len(train_ids), "evaluation_recordings": len(test_ids),
              "evaluation_hydrate_wells": sorted({items[i]["well"] for i in test_ids if items[i]["class"] == 8}),
              "folds": audit, "selected_policy": winner, "validation_baseline": baseline_validation,
              "validation_selected": winner_score, "candidates": trials,
              "stages": [
                  {"id": "raw_baseline", "name": "Raw-sensor baseline", "metrics": _score_frames([raw_frames[i] for i in test_ids], raw_p, DEFAULT_POLICY)},
                  {"id": "quality", "name": "Quality-aware model", "metrics": _score_frames(evaluation_frames, quality_p, DEFAULT_POLICY)},
                  {"id": "selected", "name": "Validation-selected alarm policy", "metrics": _score_frames(evaluation_frames, quality_p, winner)}],
              "recordings": [{"file": items[i]["name"], "well_id": items[i]["well"], **recording_metrics(frame, p, winner)} for i, frame, p in zip(test_ids, evaluation_frames, quality_p)],
              "limits": ["Previously explored, bounded 31-record/21-well subset; this is internal held-out-well evaluation, not untouched external validation.",
                         "Model scores are uncalibrated. QGL measures gas-lift injection, not oil production.",
                         "Promotion requires no increase in missed events or false-alarm minutes against the validation incumbent, with a strict improvement in at least one. Episode count, detection delay, and shorter persistence break candidate ties. Illustrative cost is descriptive only.",
                         "All four held-out hydrate recordings belong to the SAME well, WELL-00019. Four detected recordings are not four independent hydrate wells.",
                         "Raw multiclass confusion and threshold/persistence alarm metrics are different measures.",
                         "An alarm already open at event onset is reported separately and is not credited as new early warning.",
                         "Native observations are aggregated into causal one-minute bins for this subset prototype; the source Parquets remain unchanged."]}
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = BUNDLE_PATH.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(BUNDLE_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def load_bundle(path=None):
    """Load a locally generated trusted bundle; never load untrusted pickle files."""
    path = BUNDLE_PATH if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError("Fleet model not trained. Run scripts/train_fleet_model.py.")
    with path.open("rb") as stream:
        bundle = pickle.load(stream)
    if bundle.get("schema_version") != SCHEMA_VERSION or bundle.get("sensors") != list(SENSORS):
        raise ValueError("Fleet model schema mismatch; retrain the bundle")
    if set(bundle["training_wells"]) & set(bundle["excluded_wells"]):
        raise ValueError("Fleet bundle contains a demo-well training overlap")
    if sorted(bundle["excluded_wells"]) != EXCLUDED_WELLS:
        raise ValueError("Fleet bundle has incompatible demo wells")
    manifest = RAW / "manifest.json"
    if manifest.exists() and hashlib.sha256(manifest.read_bytes()).hexdigest() != bundle["manifest_sha256"]:
        raise ValueError("Fleet bundle dataset manifest changed; retrain the bundle")
    return bundle
