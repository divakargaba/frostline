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
FEATURE_VERSION = "causal-relative-v1"
MODEL_CANDIDATES = [
    {"id": "base_class", "feature_family": "base", "weighting": "class"},
    {"id": "base_record", "feature_family": "base", "weighting": "record_class"},
    {"id": "relative_record", "feature_family": "relative", "weighting": "record_class"},
    {"id": "hybrid_record", "feature_family": "hybrid", "weighting": "record_class"},
]
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


def _relative_features(frame):
    """Changes from the preceding observed hour, without fitting to a whole well.

    Pressure scales have a 1-bar numerical floor, not an operating limit.
    Celsius differences are never divided by a Celsius reference temperature.
    Only derived features are bounded; original sensor observations are retained.
    """
    core = frame.reindex(columns=[*PRESSURES, "T-TPT"])
    output = {}
    for minutes in [10, 30, 60]:
        # closed=left excludes the current sample and works with timestamp gaps.
        past = core.rolling(f"{minutes}min", closed="left", min_periods=3)
        mean, std = past.mean(), past.std()
        lag = core.shift(freq=pd.Timedelta(minutes=minutes)).reindex(core.index)
        for name in core:
            scale = mean[name].abs().clip(lower=1.) if name.startswith("P-") else 1.
            delta = core[name] - mean[name]
            prefix = f"rel_{name}_{minutes}"
            output[f"{prefix}_deviation"] = delta / scale
            output[f"{prefix}_variation"] = std[name] / scale
            output[f"{prefix}_z"] = delta / std[name].clip(lower=mean[name].abs().clip(lower=1.) * .002 if name.startswith("P-") else .01)
            output[f"{prefix}_change"] = (core[name] - lag[name]) / scale
        # Opposing upstream/downstream movement differs from a common pressure rise.
        output[f"rel_line_divergence_{minutes}"] = (
            output[f"rel_P-TPT_{minutes}_deviation"] - output[f"rel_P-MON-CKP_{minutes}_deviation"])
    for upstream, downstream, name in [("P-TPT", "P-MON-CKP", "line"), ("P-PDG", "P-TPT", "well")]:
        scale = core[upstream].abs().clip(lower=1.)
        output[f"rel_{name}_pressure_difference"] = (core[upstream] - core[downstream]) / scale
        output[f"rel_{name}_pressure_ratio"] = core[downstream] / scale
    return pd.DataFrame(output, index=frame.index).replace([np.inf, -np.inf], np.nan).clip(-100., 100.)


def feature_frame(frame, bundle=None):
    """Exact batch/stream feature builder for prepared minute sensor frames.

    Selects only sensor columns; annotations are never features. The longest
    lookback is 60 past minutes plus the current sample. Callers with injected
    values must sanitize them first, as infer_window does. Old bundles use base.
    """
    family = (bundle or {}).get("feature_family", "base")
    sensor = frame.reindex(columns=SENSORS).astype(float)
    if family == "base":
        return causal_features(sensor)
    if family == "relative":
        return _relative_features(sensor)
    if family == "hybrid":
        return pd.concat([causal_features(sensor), _relative_features(sensor)], axis=1)
    raise ValueError(f"Unknown fleet feature family: {family}")


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
    feature_history = sensors.loc[sensors.index >= sensors.index[-1] - pd.Timedelta(minutes=60)]
    features = feature_frame(feature_history, bundle).reindex(columns=bundle["feature_names"])
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
    """Batch equivalent of advance_alarm, without a Timestamp per observation."""
    policy = DEFAULT_POLICY if policy is None else policy
    available = frame.reindex(columns=PRESSURES).notna().any(axis=1).to_numpy()
    gaps = np.r_[False, np.diff(frame.index.to_numpy(dtype="datetime64[ns]")) != np.timedelta64(1, "m")]
    active, activation, recovery = False, 0, 0
    alarms = np.zeros(len(frame), dtype=bool)
    for i, (probability, valid, gap) in enumerate(zip(probabilities, available, gaps)):
        if gap:
            activation = recovery = 0
        if not valid or probability is None or not np.isfinite(probability):
            activation = recovery = 0
            continue
        if active:
            recovery = recovery + 1 if probability < policy["recovery_threshold"] else 0
            if recovery >= policy["recovery_minutes"]:
                active, activation, recovery = False, 0, 0
        else:
            activation = activation + 1 if probability >= policy["activation_threshold"] else 0
            if activation >= policy["persistence_minutes"]:
                active, recovery = True, 0
        alarms[i] = active
    return alarms


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
    matrix = np.asarray(total["classification_confusion"], dtype=float)
    true_count, predicted_count = matrix.sum(axis=1), matrix.sum(axis=0)
    recall = np.divide(matrix.diagonal(), true_count, out=np.zeros(3), where=true_count > 0)
    f1 = np.divide(2 * matrix.diagonal(), true_count + predicted_count,
                   out=np.zeros(3), where=(true_count + predicted_count) > 0)
    total["classification_recall"] = dict(zip(CLASS_NAMES, recall.tolist()))
    total["classification_macro_f1"] = float(f1.mean())
    return total


def _model(weighting="class"):
    return LGBMClassifier(n_estimators=80, num_leaves=15, learning_rate=.05,
                          min_child_samples=30, class_weight="balanced" if weighting == "class" else None, random_state=42,
                          n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)


def record_class_weights(labels):
    """Equal class mass, then equal recording mass within each class.

    Unknown labels receive zero weight but remain in the feature history.
    This does not subsample long records or use the evaluation partition.
    """
    weights = [pd.Series(0., index=label.index) for label in labels]
    for value in range(3):
        present = [index for index, label in enumerate(labels) if label.eq(value).any()]
        for index in present:
            mask = labels[index].eq(value)
            weights[index].loc[mask] = 1. / (len(present) * int(mask.sum()))
    combined = pd.concat(weights, ignore_index=True)
    known = combined.gt(0)
    if known.any():
        combined *= int(known.sum()) / combined.sum()
    return combined


def _fit(frames, spec=None, prepared_features=None):
    spec = MODEL_CANDIDATES[0] if spec is None else spec
    x = pd.concat(prepared_features if prepared_features is not None else
                  [feature_frame(frame, spec) for frame in frames], ignore_index=True)
    y = pd.concat([targets(frame) for frame in frames], ignore_index=True)
    known = y.notna()
    if set(y[known]) != {0, 1, 2}:
        raise ValueError("Training partition must contain all three target classes")
    weights = record_class_weights([targets(frame) for frame in frames])[known] if spec["weighting"] == "record_class" else None
    model = _model(spec["weighting"]).fit(x[known], y[known].astype(int), sample_weight=weights)
    return model, list(x.columns)


def _predict(model, frame, spec=None):
    return model.predict_proba(feature_frame(frame, spec))


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


def model_candidate_qualifies(candidate, incumbent):
    """Predeclared model gate: neither operational nor classification regressions."""
    return (candidate["events_detected"] >= incumbent["events_detected"]
            and candidate["false_alarm_minutes"] <= incumbent["false_alarm_minutes"]
            and candidate["classification_recall"]["lookalike"] >= incumbent["classification_recall"]["lookalike"]
            and candidate["classification_macro_f1"] >= incumbent["classification_macro_f1"]
            and (candidate_qualifies(candidate, incumbent)
                 or candidate["classification_recall"]["lookalike"] > incumbent["classification_recall"]["lookalike"]
                 or candidate["classification_macro_f1"] > incumbent["classification_macro_f1"]))


def _model_candidate_key(score, policy, candidate_index):
    return (-score["classification_recall"]["lookalike"], -score["classification_macro_f1"],
            *_candidate_key(score, policy), candidate_index)


def _policy_trials(frames, probabilities):
    trials = []
    for threshold in [.5, .6, .7, .8, .9]:
        for persistence in [3, 5, 10]:
            policy = {"activation_threshold": threshold, "persistence_minutes": persistence,
                      "recovery_threshold": round(threshold - .1, 2), "recovery_minutes": 5}
            trials.append({"policy": policy, "metrics": _score_frames(frames, probabilities, policy)})
    return trials


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
    splits = list(GroupKFold(3).split(train_ids, groups=groups))
    audit = [{"fold": fold, "train_wells": sorted(set(groups[a])), "validation_wells": sorted(set(groups[b]))}
             for fold, (a, b) in enumerate(splits, 1)]
    comparison, trials_by_model = [], {}
    winner_spec, winner, winner_score = MODEL_CANDIDATES[0], None, None
    # This complete, bounded search finishes before any held-out prediction.
    for candidate_index, spec in enumerate(MODEL_CANDIDATES):
        features = [feature_frame(frame, spec) for frame in train_frames]
        oof = [None] * len(train_ids)
        for fold, (a, b) in enumerate(splits, 1):
            print(f"Fitting {spec['id']} grouped validation fold {fold}/3", flush=True)
            fold_model, _ = _fit([train_frames[j] for j in a], spec, [features[j] for j in a])
            for j in b:
                oof[j] = fold_model.predict_proba(features[j])
        trials = _policy_trials(train_frames, oof)
        trials_by_model[spec["id"]] = trials
        if candidate_index == 0:
            baseline_validation = trials[0]["metrics"]
            incumbent_policy, incumbent_score = dict(DEFAULT_POLICY), baseline_validation
            for trial in trials:
                if (candidate_qualifies(trial["metrics"], baseline_validation)
                        and _candidate_key(trial["metrics"], trial["policy"]) < _candidate_key(incumbent_score, incumbent_policy)):
                    incumbent_policy, incumbent_score = trial["policy"], trial["metrics"]
            winner, winner_score = incumbent_policy, incumbent_score
            chosen = {"policy": incumbent_policy, "metrics": incumbent_score}
            qualifies = True
        else:
            eligible = [trial for trial in trials if model_candidate_qualifies(trial["metrics"], incumbent_score)]
            chosen = min(eligible or trials, key=lambda trial: _model_candidate_key(trial["metrics"], trial["policy"], candidate_index))
            qualifies = bool(eligible)
            if qualifies and _model_candidate_key(chosen["metrics"], chosen["policy"], candidate_index) < _model_candidate_key(winner_score, winner, MODEL_CANDIDATES.index(winner_spec)):
                winner_spec, winner, winner_score = spec, chosen["policy"], chosen["metrics"]
        comparison.append({**spec, "policy": chosen["policy"], "validation": chosen["metrics"],
                           "qualifies": qualifies, "trials": trials})
        print(json.dumps({"candidate": spec["id"], "qualifies": qualifies,
                          "events": chosen["metrics"]["events_detected"],
                          "false_alarm_minutes": chosen["metrics"]["false_alarm_minutes"],
                          "lookalike_recall": chosen["metrics"]["classification_recall"]["lookalike"],
                          "macro_f1": chosen["metrics"]["classification_macro_f1"]}), flush=True)
    for entry in comparison:
        entry["selected"] = entry["id"] == winner_spec["id"]
    trials = trials_by_model[winner_spec["id"]]
    for trial in trials:
        trial.update(qualifies=model_candidate_qualifies(trial["metrics"], incumbent_score)
                     if winner_spec["id"] != "base_class" else candidate_qualifies(trial["metrics"], baseline_validation),
                     accepted=trial["policy"] == winner,
                     reason="Model and policy selection uses grouped validation only; the frozen incumbent remains unless every predeclared non-regression gate passes.")
    print(f"Selection frozen: {winner_spec['id']} {winner}. Fitting final models before held-out reporting.", flush=True)
    model, feature_names = _fit(train_frames, winner_spec)
    quality_model = model if winner_spec["id"] == "base_class" else _fit(train_frames)[0]
    raw_frames = [minute_frame(RAW / str(items[i]["class"]) / items[i]["name"]) for i in range(len(items))]
    baseline_model, _ = _fit([raw_frames[i] for i in train_ids])
    evaluation_frames = [frames[i] for i in test_ids]
    quality_p = [_predict(quality_model, frame) for frame in evaluation_frames]
    selected_p = quality_p if winner_spec["id"] == "base_class" else [_predict(model, frame, winner_spec) for frame in evaluation_frames]
    raw_p = [_predict(baseline_model, raw_frames[i]) for i in test_ids]
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    identity = json.dumps({"schema": SCHEMA_VERSION, "feature_version": FEATURE_VERSION, "candidate": winner_spec,
                           "manifest": manifest_hash, "training_wells": training_wells, "policy": winner,
                           "model_parameters": model.get_params()}, sort_keys=True)
    model_id = "fleet-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
    bundle = {"model": model, "model_id": model_id, "schema_version": SCHEMA_VERSION,
              "feature_version": FEATURE_VERSION, "feature_family": winner_spec["feature_family"],
              "weighting": winner_spec["weighting"], "selected_candidate": winner_spec["id"],
              "feature_names": feature_names, "sensors": list(SENSORS), "class_names": CLASS_NAMES,
              "policy": winner, "policy_version": model_id, "training_wells": training_wells,
              "excluded_wells": EXCLUDED_WELLS, "manifest_sha256": manifest_hash,
              "source_revision": manifest["revision"], "license": manifest["license"],
              "training_recordings": [items[i]["name"] for i in train_ids]}
    report = {"model_id": model_id, "schema_version": SCHEMA_VERSION, "manifest_sha256": manifest_hash,
              "protocol": "Fixed four demo wells excluded completely; 3-fold grouped validation on the other 17 wells; four predeclared feature/weighting families with fixed LightGBM capacity; 15 threshold/persistence candidates per family; five-minute recovery hysteresis. All model and policy choices frozen before held-out prediction.",
              "training_wells": training_wells, "excluded_wells": EXCLUDED_WELLS,
              "training_recordings": len(train_ids), "evaluation_recordings": len(test_ids),
              "evaluation_hydrate_wells": sorted({items[i]["well"] for i in test_ids if items[i]["class"] == 8}),
              "folds": audit, "selected_policy": winner, "validation_baseline": baseline_validation,
              "validation_selected": winner_score, "candidates": trials,
              "selected_candidate": winner_spec, "feature_version": FEATURE_VERSION,
              "validation_incumbent": {"candidate": "base_class", "policy": incumbent_policy, "metrics": incumbent_score},
              "model_comparison": comparison,
              "selection_rules": {"candidate_count": len(MODEL_CANDIDATES), "policies_per_candidate": 15,
                                  "non_regression_gates": ["hydrate events detected", "false-alarm minutes", "lookalike classification recall", "classification macro F1"],
                                  "strict_improvement_required": True,
                                  "rank": ["higher lookalike recall", "higher macro F1", "more events detected", "fewer false-alarm minutes", "fewer episodes", "shorter mean detection delay", "shorter persistence", "lower threshold", "simpler candidate"],
                                  "heldout_used_for_selection": False},
              "stages": [
                  {"id": "raw_baseline", "name": "Raw-sensor baseline", "metrics": _score_frames([raw_frames[i] for i in test_ids], raw_p, DEFAULT_POLICY)},
                  {"id": "quality", "name": "Quality-aware model", "metrics": _score_frames(evaluation_frames, quality_p, DEFAULT_POLICY)},
                  {"id": "selected", "name": "Validation-selected model and alarm policy", "metrics": _score_frames(evaluation_frames, selected_p, winner)}],
              "previous_selected_heldout": _score_frames(evaluation_frames, quality_p, incumbent_policy),
              "recordings": [{"file": items[i]["name"], "well_id": items[i]["well"], **recording_metrics(frame, p, winner)} for i, frame, p in zip(test_ids, evaluation_frames, selected_p)],
              "limits": ["Previously explored, bounded 31-record/21-well subset; this is internal held-out-well evaluation, not untouched external validation.",
                         "Model scores are uncalibrated. QGL measures gas-lift injection, not oil production.",
                         "Policy selection retains the original event/false-minute Pareto gate. Model promotion additionally requires no reduction in lookalike recall or macro F1, with a strict improvement in at least one of these four measures. Illustrative cost is descriptive only.",
                         "Four fixed model families and 15 policies each were compared on the same development folds. Their validation scores are selection results, not unbiased estimates of deployment performance.",
                         "Relative features use at most 60 preceding observed minutes. Per-record/class weights equalize class mass and recordings within each class; they do not discard observations or balance using held-out data.",
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
    if bundle.get("feature_family", "base") not in {"base", "relative", "hybrid"}:
        raise ValueError("Fleet bundle feature family is unsupported")
    if bundle.get("feature_version", FEATURE_VERSION) != FEATURE_VERSION:
        raise ValueError("Fleet bundle feature version is unsupported; retrain the bundle")
    if set(bundle["training_wells"]) & set(bundle["excluded_wells"]):
        raise ValueError("Fleet bundle contains a demo-well training overlap")
    if sorted(bundle["excluded_wells"]) != EXCLUDED_WELLS:
        raise ValueError("Fleet bundle has incompatible demo wells")
    manifest = RAW / "manifest.json"
    if manifest.exists() and hashlib.sha256(manifest.read_bytes()).hexdigest() != bundle["manifest_sha256"]:
        raise ValueError("Fleet bundle dataset manifest changed; retrain the bundle")
    return bundle
