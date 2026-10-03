"""Fixed-protocol 3W pilot with predictions from models that never saw the well.

This is external validation of a separate three-class classifier, NOT a claim
that a pressure-drop rule generalizes to real hydrate. No thresholds are tuned
on these folds; no thermodynamics, treatment or forecast is inferred.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from src.load import SENSORS
from src.research import ROOT

OUT = ROOT / "data/processed/research"
RAW = ROOT / "data/raw/research"


def minute_frame(path):
    raw = pd.read_parquet(path)
    if "timestamp" in raw:
        raw = raw.set_index("timestamp")
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    values = raw.reindex(columns=SENSORS).astype(float)
    for col in values:
        if col.startswith("P-"):
            values[col] /= 1e5
    # A bin is timestamped WHEN it is fully available, at its right edge.
    frame = values.resample("1min", label="right", closed="left").mean()
    labels = raw["class"].resample("1min", label="right", closed="left").agg(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan)
    frame["class"] = labels
    return frame


def causal_features(frame):
    """No whole-file frozen filtering, label-based dropping or interpolation."""
    base = frame.reindex(columns=SENSORS).copy()
    base["line_dp"] = base["P-TPT"] - base["P-MON-CKP"]
    base["well_dp"] = base["P-PDG"] - base["P-TPT"]
    parts = [base]
    for minutes in [10, 60]:
        roll = base.rolling(f"{minutes}min", min_periods=3)
        parts.extend([roll.mean().add_suffix(f"_mean{minutes}"), roll.std().add_suffix(f"_std{minutes}"), (base - roll.mean()).add_suffix(f"_delta{minutes}")])
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def targets(frame):
    return frame["class"].map({0: 0, 8: 1, 108: 1, 6: 2, 106: 2, 7: 2, 107: 2})


def alarms(probability, frame):
    available = frame[["P-PDG", "P-TPT", "P-MON-CKP"]].notna().any(axis=1)
    evidence = pd.Series(np.asarray(probability) >= .5, index=frame.index) & available
    # Frame is a complete minute grid, so missing ticks interrupt the run.
    return evidence.rolling(3, min_periods=3).sum().eq(3).to_numpy()


def event_metrics(frame, prediction, probabilities):
    y = targets(frame)
    valid = y.notna().to_numpy()
    alarm = alarms(probabilities[:, 1], frame)
    truth_normal = y.eq(0).to_numpy()
    false = alarm & truth_normal
    false_starts = int((false & ~np.r_[False, false[:-1]]).sum())
    forming = frame.index[frame["class"].eq(108)]
    established = frame.index[frame["class"].eq(8)]
    hydrate = y.eq(1).to_numpy()
    event_alarms = frame.index[alarm & hydrate]
    first = event_alarms[0] if len(event_alarms) else None
    est = established[0] if len(established) else None
    lead = (est - first).total_seconds() / 60 if first is not None and est is not None else None
    return {"minutes": int(valid.sum()), "normal_minutes": int(truth_normal.sum()),
            "hydrate_minutes": int(hydrate.sum()), "missing_pressure_minutes": int(frame[["P-PDG", "P-TPT", "P-MON-CKP"]].isna().all(axis=1).sum()),
            "false_alarm_episodes": false_starts, "detected": bool(len(event_alarms)), "lead_minutes": lead,
            "early": bool(lead is not None and lead > 0), "late": bool(lead is not None and lead <= 0),
            "has_established": bool(len(established)), "has_forming": bool(len(forming)),
            "misdiagnosed": bool((alarm & y.eq(2).to_numpy()).any()),
            "first_alarm": first.isoformat() if first is not None else None,
            "t_form": forming[0].isoformat() if len(forming) else None, "t_est": est.isoformat() if est is not None else None}


def replay_frames(frame, probabilities, fold):
    alarm = alarms(probabilities[:, 1], frame)
    output = []
    last_alert = -1000
    for i, (timestamp, row) in enumerate(frame.iterrows()):
        available = sum(pd.notna(row[s]) for s in ["P-PDG", "P-TPT", "P-MON-CKP"])
        p = probabilities[i]
        decision = "WATCH" if not available else "ALERT" if alarm[i] else "WATCH" if p[1] >= .3 else "DISMISS"
        diagnosis = "Missing pressure evidence" if not available else "Hydrate-like pattern" if decision == "ALERT" else "Restriction / scaling pattern" if p[2] >= .5 else "Evidence below alarm threshold"
        notify = decision == "ALERT" and i - last_alert >= 60
        if notify:
            last_alert = i
        truth = {0: "Normal", 8: "Established hydrate", 108: "Forming hydrate", 6: "Quick restriction", 106: "Forming restriction", 7: "Scaling", 107: "Forming scaling"}.get(row["class"], "Unlabelled")
        output.append({"t": timestamp.isoformat(), "decision": decision, "diagnosis": diagnosis,
                       "brief": "Investigate with an operator. Model scores are not calibrated probabilities or a treatment recommendation." if decision == "ALERT" else "Recheck the next minute; preserve alternative explanations and sensor quality in the record.",
                       "notify": notify, "recheck_minutes": 1, "quality": "good" if available == 3 else "degraded",
                       "ground_truth": truth, "scores": {"normal": float(p[0]), "hydrate": float(p[1]), "lookalike": float(p[2])},
                       "sensors": {"pressure_bar": float(row["P-TPT"]) if pd.notna(row["P-TPT"]) else None, "temp_C": float(row["T-TPT"]) if pd.notna(row["T-TPT"]) else None, "downstream_bar": float(row["P-MON-CKP"]) if pd.notna(row["P-MON-CKP"]) else None},
                       "evidence": [
                           {"tool": "quality_check", "summary": f"{available}/3 pressure sensors available. Missing values retained; no future imputation."},
                           {"tool": "classify_window", "summary": f"Held-out-well fold {fold}: hydrate score {p[1]:.3f}, look-alike {p[2]:.3f}, normal {p[0]:.3f}."},
                           {"tool": "confirm_persistence", "summary": "Score must be ≥0.50 for 3 consecutive minutes. " + ("Confirmed." if alarm[i] else "Not confirmed.")},
                           {"tool": "schedule_followup", "summary": "Recheck in 1 minute. " + ("Notify operator." if notify else "Notification cooldown: 60 minutes.")},
                       ]})
    # Playback thinning only, AFTER full-resolution predictions and evaluation.
    stride = max(1, int(np.ceil(len(output) / 800)))
    indices = set(range(0, len(output), stride)) | {len(output) - 1}
    indices.update(i for i in range(1, len(output)) if output[i]["decision"] != output[i - 1]["decision"] or output[i]["ground_truth"] != output[i - 1]["ground_truth"])
    return [output[i] for i in sorted(indices)], stride


def run():
    manifest = json.loads((RAW / "manifest.json").read_text())
    frames, features, label_series, groups = [], [], [], []
    for item in manifest["files"]:
        path = RAW / str(item["class"]) / item["name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Data checksum mismatch: {path.name}")
        frame = minute_frame(path)
        frames.append(frame)
        features.append(causal_features(frame))
        label_series.append(targets(frame))
        groups.extend([item["well"]] * len(frame))
    x = pd.concat(features, ignore_index=True)
    y = pd.concat(label_series, ignore_index=True)
    groups = np.asarray(groups)
    known = y.notna().to_numpy()
    # Fixed BEFORE any outcomes: 3 folds, 80 trees, 15 leaves, 0.5 threshold.
    probabilities = np.zeros((len(x), 3))
    folds = np.zeros(len(x), dtype=int)
    audits, importance = [], np.zeros(x.shape[1])
    for fold, (train, test) in enumerate(GroupKFold(n_splits=3).split(x, groups=groups), 1):
        train = train[known[train]]
        train_wells, test_wells = sorted(set(groups[train])), sorted(set(groups[test]))
        assert not set(train_wells) & set(test_wells)
        if set(y.iloc[train].unique()) != {0, 1, 2}:
            raise ValueError("Training fold missing a class; change data coverage, not test labels")
        model = LGBMClassifier(n_estimators=80, num_leaves=15, max_depth=-1, learning_rate=.05, min_child_samples=30, class_weight="balanced", random_state=42, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
        model.fit(x.iloc[train], y.iloc[train].astype(int))
        probabilities[test] = model.predict_proba(x.iloc[test])
        folds[test] = fold
        importance += model.feature_importances_
        audits.append({"fold": fold, "train_wells": train_wells, "test_wells": test_wells, "train_minutes": len(train), "test_minutes": int(known[test].sum()), "overlap": 0})
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y[known], predictions[known], labels=[0, 1, 2])
    recordings, replays = [], {}
    offset = 0
    for item, frame in zip(manifest["files"], frames):
        n = len(frame)
        p = probabilities[offset:offset + n]
        fold = int(folds[offset])
        scores = event_metrics(frame, predictions[offset:offset + n], p)
        recordings.append({"file": item["name"], "well": item["well"], "class": item["class"], "fold": fold, **scores})
        # First lexical example of each source class; never cherry-pick successes.
        category = {0: "normal", 6: "restriction", 7: "scaling", 8: "hydrate"}[item["class"]]
        key = f"real-{category}"
        if key not in replays:
            events, stride = replay_frames(frame, p, fold)
            replays[key] = {"id": key, "title": f"3W · {category.capitalize()}", "provenance": "Real recording · held-out well",
                            "description": f"{item['name']} · fold {fold}. First file in lexical order for this class. Predictions made every minute; replay displays every {stride} minutes plus decision/phase changes.",
                            "cadence_minutes": 1, "display_stride": stride, "policy_id": f"3w-group-fold-{fold}", "frames": events,
                            "sensor_labels": {"pressure_bar": "Upstream P-TPT · bar", "temp_C": "T-TPT · °C", "downstream_bar": "Downstream P-MON · bar"}}
        offset += n
    hydrate_records = [r for r in recordings if r["class"] == 8]
    look_records = [r for r in recordings if r["class"] in [6, 7]]
    normal_minutes = sum(r["normal_minutes"] for r in recordings)
    result = {"status": "complete", "protocol": "3-fold GroupKFold; entire wells held out; fixed LightGBM (80 trees, 15 leaves); no threshold tuning; hydrate alarm ≥0.5 for 3 minutes.",
              "revision": manifest["revision"], "license": manifest["license"], "selection": manifest["selection"],
              "recordings": recordings, "wells": len(set(groups)), "minutes": int(known.sum()), "total_minutes": len(x),
              "class_names": ["Normal", "Hydrate", "Look-alike"], "confusion_matrix": matrix.tolist(),
              "macro_f1": float(f1_score(y[known], predictions[known], average="macro")),
              "hydrate_recall": float(matrix[1, 1] / matrix[1].sum()) if matrix[1].sum() else None,
              "events_detected": sum(r["detected"] for r in hydrate_records), "hydrate_recordings": len(hydrate_records),
              "early_events": sum(r["early"] for r in hydrate_records), "established_recordings": sum(r["has_established"] for r in hydrate_records),
              "lookalikes_flagged": sum(r["misdiagnosed"] for r in look_records), "lookalike_recordings": len(look_records),
              "false_alarms_per_normal_day": sum(r["false_alarm_episodes"] for r in recordings) / (normal_minutes / 1440) if normal_minutes else None,
              "folds": audits, "feature_importance": [{"name": x.columns[i], "splits": int(importance[i])} for i in np.argsort(-importance)[:10]],
              "limits": ["A bounded pilot, not the full 3W benchmark. Only production-line hydrate (8), normal (0), restriction (6) and scaling (7).",
                         "Class and well identity are correlated. Whole-well separation helps, but does not eliminate sampling bias or sensor-availability shortcuts.",
                         "Minute labels use within-minute mode; timestamps mark the end of the sensor aggregation interval.",
                         "Cross-validation produces out-of-fold predictions, not a separate untouched external test set. No parameter tuning used these scores.",
                         "Model scores are not calibrated confidence. No deployment reliability, gas composition, chemical dosing or blockage ETA is inferred.",
                         "Real-data quality gating currently checks pressure availability. Frozen-sensor injection checks are demonstrated separately on the seed workflow."]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real_report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (OUT / "real_replays.json").write_text(json.dumps(replays, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ["recordings", "folds", "confusion_matrix", "feature_importance", "limits"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
