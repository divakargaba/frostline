"""Reproducible seed experiment and bounded autonomous policy selection.

Labels are accepted by the evaluator/selector only. predict() and investigate()
receive sensor values, timestamps and a frozen policy. No LLM or remote key.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data/seed/well_hydrate_seed.csv"
SENSORS = ["pressure_bar", "temp_C", "flow_Lps"]
SOURCES = [
    {"title": "Case 9 · official problem & starter", "url": "https://github.com/nagusubra/industry-hackathon-lab/tree/main/01-energy-and-infrastructure-systems/Case%209%20-%20Autonomous%20Offshore%20Well%20Event%20Flag%20Agent"},
    {"title": "Judging rubric", "url": "https://github.com/nagusubra/industry-hackathon-lab/blob/main/JUDGING_RUBRIC.md"},
    {"title": "Petrobras 3W dataset · CC BY 4.0", "url": "https://github.com/petrobras/3W"},
    {"title": "3W dataset paper · sensor behavior and labels", "url": "https://arxiv.org/html/2507.01048v1"},
    {"title": "Threshold tuning · separate validation data", "url": "https://scikit-learn.org/stable/modules/classification_threshold.html"},
    {"title": "GroupKFold · hold out entire wells", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html"},
    {"title": "NIST · EWMA for gradual changes", "url": "https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm"},
    {"title": "Anomaly evaluation · pitfalls of point adjustment", "url": "https://arxiv.org/abs/2109.05257"},
]


@dataclass(frozen=True)
class Policy:
    pressure_percentile: int = 10
    confirmation_percentile: int | None = 20
    consecutive: int = 1
    smoothing: bool = False
    pressure_cutoff: float = 0
    temperature_cutoff: float | None = None
    flow_cutoff: float | None = None


def load_seed(path=SEED):
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.timestamp.duplicated().any():
        raise ValueError("Seed timestamps must be unique")
    if not df.label.isin([0, 1]).all():
        raise ValueError("Seed labels must be binary")
    # Preserve missing rows: dropping them would turn separated ticks into runs.
    return df


def split_seed(df):
    start = df.timestamp.min()
    cal = df[df.timestamp < start + pd.Timedelta(days=10)].copy()
    val = df[(df.timestamp >= start + pd.Timedelta(days=10)) & (df.timestamp < start + pd.Timedelta(days=20))].copy()
    test = df[df.timestamp >= start + pd.Timedelta(days=20)].copy()
    if any(part.empty for part in [cal, val, test]):
        raise ValueError("Need calibration, validation and final test periods")
    return tuple(p.reset_index(drop=True) for p in [cal, val, test])


def fit_policy(df, pressure=10, confirmation=20, consecutive=1, smoothing=False):
    return Policy(pressure, confirmation, consecutive, smoothing,
                  float(df.pressure_bar.quantile(pressure / 100)),
                  float(df.temp_C.quantile(confirmation / 100)) if confirmation else None,
                  float(df.flow_Lps.quantile(confirmation / 100)) if confirmation else None)


def predict(sensors, policy):
    """One causal decision per hourly row; missing readings/gaps break persistence."""
    p = sensors.pressure_bar
    if policy.smoothing:
        # Reset smoothing across missing pressure or timestamp gaps.
        breaks = p.isna() | p.shift().isna() | sensors.timestamp.diff().ne(pd.Timedelta(hours=1))
        p = p.groupby(breaks.cumsum()).transform(lambda x: x.ewm(span=3, adjust=False).mean())
    evidence = p.lt(policy.pressure_cutoff) & sensors.pressure_bar.notna()
    if policy.confirmation_percentile:
        evidence &= sensors.temp_C.lt(policy.temperature_cutoff) | sensors.flow_Lps.lt(policy.flow_cutoff)
    groups = sensors.timestamp.diff().ne(pd.Timedelta(hours=1)).cumsum()
    result = evidence.groupby(groups).transform(lambda x: x.rolling(policy.consecutive, min_periods=policy.consecutive).sum().eq(policy.consecutive))
    return result.astype(bool)


def metrics(df, predictions):
    """No point adjustment. Bad-hour recall and event coverage stay separate."""
    frame = df.reset_index(drop=True)
    pred = np.asarray(predictions, dtype=bool)
    valid = frame.pressure_bar.notna().to_numpy()
    target = frame.label.eq(1).to_numpy()
    tp = int((pred & target & valid).sum())
    fp = int((pred & ~target & valid).sum())
    fn = int((~pred & target & valid).sum())
    tn = int((~pred & ~target & valid).sum())
    adjacent = frame.timestamp.diff().eq(pd.Timedelta(hours=1)).to_numpy()
    false = pred & ~target & valid
    false_episodes = int((false & ~(np.r_[False, false[:-1]] & adjacent)).sum())
    starts = np.flatnonzero(target & ~(np.r_[False, target[:-1]] & adjacent))
    events = []
    for start in starts:
        end = start
        while end + 1 < len(frame) and target[end + 1] and adjacent[end + 1]:
            end += 1
        alarms = np.flatnonzero(pred[start:end + 1] & valid[start:end + 1])
        delay = float((frame.timestamp.iloc[start + alarms[0]] - frame.timestamp.iloc[start]).total_seconds() / 3600) if len(alarms) else None
        events.append({"start": frame.timestamp.iloc[start].isoformat(), "end": frame.timestamp.iloc[end].isoformat(), "detected": bool(len(alarms)), "delay_hours": delay, "bad_hours": int(end - start + 1)})
    delays = [e["delay_hours"] for e in events if e["detected"]]
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "bad_hours": tp + fn, "evaluated_hours": int(valid.sum()), "excluded_hours": int((~valid).sum()),
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "false_alarm_episodes": false_episodes, "false_alarms_per_normal_day": false_episodes / ((fp + tn) / 24) if fp + tn else None,
            "events_detected": sum(e["detected"] for e in events), "event_count": len(events),
            "delay_hours": float(np.mean(delays)) if delays else None, "events": events,
            "cost": 5 * fn + fp}


def select_policy(calibration, validation):
    """The bounded search never receives the final test period."""
    trials = []
    for p in [5, 10, 15, 20]:
        for c in [10, 20, 30]:
            for n in [1, 2]:
                policy = fit_policy(calibration, p, c, n)
                m = metrics(validation, predict(validation[["timestamp", *SENSORS]], policy))
                trials.append({"id": len(trials) + 1, "policy": asdict(policy), "metrics": m,
                               "name": f"P{p} + confirmation P{c} · {n}h"})
    def key(t):
        m = t["metrics"]
        return (m["cost"], m["false_positive"], m["delay_hours"] if m["delay_hours"] is not None else float("inf"), t["id"])
    winner = min(trials, key=key)
    incumbent = metrics(validation, predict(validation[["timestamp", *SENSORS]], fit_policy(calibration, 10, None)))
    # Only promote a strict improvement; ties retain the incumbent.
    promoted = winner["metrics"]["cost"] < incumbent["cost"]
    config = winner["policy"] if promoted else asdict(fit_policy(calibration, 10, None))
    best = incumbent["cost"]
    for trial in trials:
        improved = trial["metrics"]["cost"] < best
        trial["action"] = "candidate improved" if improved else "rejected"
        best = min(best, trial["metrics"]["cost"])
        trial["selected"] = promoted and trial["id"] == winner["id"]
    return {"trials": trials, "selected_config": config, "selected_trial": winner["id"] if promoted else None,
            "promoted": promoted, "incumbent_cost": incumbent["cost"], "validation_cost": winner["metrics"]["cost"] if promoted else incumbent["cost"],
            "objective": "5 × missed bad hours + false-alarm hours", "weight_note": "Illustrative error weights; not a validated operating cost.",
            "reason": "Promote only if validation cost beats the P10 pressure incumbent. Refit chosen percentiles on days 1–20, then freeze before final testing."}


VARIANTS = [
    ("B0", "Always normal", "No event detection; exposes misleading accuracy.", None),
    ("B1", "Official starter · P5", "Pressure below the training 5th percentile.", (5, None, 1, False)),
    ("B2", "More sensitive · P10", "Change only the pressure threshold.", (10, None, 1, False)),
    ("M1", "Confirm with a second sensor", "P10 pressure + low temperature OR low flow (P20).", (10, 20, 1, False)),
    ("M2", "Require persistence", "Based on M1: require two consecutive hourly samples.", (10, 20, 2, False)),
    ("M3", "Smooth pressure", "Based on M1: add a causal 3-sample pressure EWMA.", (10, 20, 1, True)),
    ("AUTO", "Autonomously selected", "Choose from 24 bounded policies using validation cost.", "auto"),
]


def experiment(df=None):
    df = load_seed() if df is None else df.copy()
    cal, val, test = split_seed(df)
    selection = select_policy(cal, val)
    chosen = selection["selected_config"]
    train = pd.concat([cal, val], ignore_index=True)
    systems = []
    for id_, name, description, config in VARIANTS:
        if config == "auto":
            config = (chosen["pressure_percentile"], chosen["confirmation_percentile"], chosen["consecutive"], chosen["smoothing"])
        policies = [fit_policy(part, *config) if config else None for part in [cal, train]]
        scores = [metrics(part, predict(part[["timestamp", *SENSORS]], policy) if policy else np.zeros(len(part), dtype=bool)) for part, policy in zip([val, test], policies)]
        systems.append({"id": id_, "name": name, "description": description, "validation": scores[0], "test": scores[1], "policy": asdict(policies[1]) if config else None})
    policy = systems[-1]["policy"]
    fingerprint = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:12]
    return {"dataset": "Official Case 9 synthetic seed", "rows": len(df), "missing_pressure": int(df.pressure_bar.isna().sum()),
            "split": [{"name": name, "days": 10, "rows": len(part), "usable": int(part.pressure_bar.notna().sum()), "start": part.timestamp.min().isoformat(), "end": part.timestamp.max().isoformat()} for name, part in zip(["Calibrate", "Select", "Test"], [cal, val, test])],
            "systems": systems, "selection": selection, "policy_id": fingerprint,
            "dataset_sha256": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
            "limits": ["Synthetic data; one incident in the final 10-day test. This cannot establish field reliability.",
                       "Delay is measured AFTER the first positive label. The seed has no forming/established phases, so this is not early-warning lead time.",
                       "Missing-pressure hours are excluded equally for starter comparability and reported explicitly. Persistence resets at missing readings or timestamp gaps.",
                       "The selector uses historical validation labels; live decisions never receive labels. Retuning is offline, not unsupervised online learning.",
                       "No validated hydrate thermodynamics, dosing, blockage forecast or LLM claims in this prototype."],
            "sources": SOURCES}


def investigate(sensors, policy):
    """Deterministic autonomous investigation. Four bounded evidence steps per tick.

    Quality gates precede diagnosis. Repeated exact readings flag a frozen sensor
    after six hours. No future readings or evaluator labels are accessed.
    """
    predicted = predict(sensors, policy).to_numpy()
    records = []
    last_notify = None
    for i, row in sensors.reset_index(drop=True).iterrows():
        history = sensors.iloc[max(0, i - 5):i + 1]
        missing = [s for s in SENSORS if pd.isna(row[s])]
        frozen = [s for s in SENSORS if len(history) == 6 and history[s].notna().all() and history[s].nunique() == 1 and history.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()]
        low_p = pd.notna(row.pressure_bar) and row.pressure_bar < policy.pressure_cutoff
        low_t = pd.notna(row.temp_C) and policy.temperature_cutoff is not None and row.temp_C < policy.temperature_cutoff
        low_f = pd.notna(row.flow_Lps) and policy.flow_cutoff is not None and row.flow_Lps < policy.flow_cutoff
        quality_block = "pressure_bar" in missing or bool(frozen) or (policy.confirmation_percentile is not None and all(s in missing for s in ["temp_C", "flow_Lps"]))
        if quality_block:
            status, diagnosis, brief = "WATCH", "Sensor quality needs attention", "Evidence is incomplete. Check telemetry and re-evaluate at the next hourly sample."
        elif predicted[i]:
            status, diagnosis, brief = "ALERT", "Hydrate-like deterioration", "The frozen rule is satisfied. Ask an operator to investigate the restriction; this pattern alone does not prove hydrate."
        elif low_p:
            status, diagnosis, brief = "WATCH", "Unconfirmed pressure anomaly", "Pressure crossed its threshold. Wait for corroboration or persistence before escalating."
        else:
            status, diagnosis, brief = "DISMISS", "No rule trigger", "Continue monitoring. Current evidence does not satisfy the selected deterioration rule."
        elapsed = (row.timestamp - last_notify).total_seconds() / 3600 if last_notify is not None else float("inf")
        notify = status == "ALERT" and elapsed >= 6
        if notify:
            last_notify = row.timestamp
        evidence = [
            {"tool": "quality_check", "summary": f"Missing: {', '.join(missing) or 'none'}. Frozen (6h): {', '.join(frozen) or 'none'}.", "result": {"missing": missing, "frozen": frozen}},
            {"tool": "compare_pressure", "summary": f"Pressure threshold {policy.pressure_cutoff:.2f} bar; crossing {'yes' if low_p else 'no'}.", "result": {"threshold_bar": policy.pressure_cutoff, "crossed": bool(low_p)}},
            {"tool": "check_corroboration", "summary": f"Low temperature: {'yes' if low_t else 'no'}. Low flow: {'yes' if low_f else 'no'}. Required duration: {policy.consecutive}h.", "result": {"low_temperature": bool(low_t), "low_flow": bool(low_f), "required_hours": policy.consecutive}},
            {"tool": "schedule_followup", "summary": "Notify operator; recheck in 60 min." if notify else "Recheck in 60 min." + (" Repeat notification suppressed by 6h cooldown." if status == "ALERT" else ""), "result": {"recheck_minutes": 60, "notify": notify}},
        ]
        records.append({"t": row.timestamp.isoformat(), "decision": status, "diagnosis": diagnosis, "brief": brief, "notify": notify,
                        "quality": "degraded" if missing or frozen else "good", "recheck_minutes": 60, "evidence": evidence,
                        "sensors": {s: None if pd.isna(row[s]) else float(row[s]) for s in SENSORS}})
    return records


def seed_scenarios(report=None):
    report = experiment() if report is None else report
    df = load_seed()
    _, _, test = split_seed(df)
    policy = Policy(**report["systems"][-1]["policy"])
    # Fault challenges use a normal section of the seed. They are never scored
    # as measured improvements and do not participate in selection or testing.
    normal = test.iloc[:48].copy().reset_index(drop=True)
    dip = normal.copy()
    dip.loc[24, "pressure_bar"] = policy.pressure_cutoff - 8
    dip.loc[24, "temp_C"] = max(normal.temp_C.max(), policy.temperature_cutoff + 5)
    dip.loc[24, "flow_Lps"] = max(normal.flow_Lps.max(), policy.flow_cutoff + 2)
    missing = normal.copy()
    missing.loc[20:28, "pressure_bar"] = np.nan
    frozen = normal.copy()
    frozen.loc[18:32, SENSORS] = normal.loc[17, SENSORS].to_numpy()
    items = [
        ("seed-test", "The held-out incident", "Synthetic · final test", test, "Full final 10 days, including the Jan 26 event. Ground truth is used only to score results."),
        ("seed-dip", "An isolated pressure dip", "Injected challenge", dip, "One pressure reading changed; temperature and flow remain above their thresholds. Tests corroboration."),
        ("seed-missing", "Pressure sensor goes offline", "Injected challenge", missing, "Nine readings removed. The agent must defer its diagnosis and flag data quality."),
        ("seed-frozen", "A frozen sensor feed", "Injected challenge", frozen, "Fifteen readings held constant. Quality gating begins at the sixth identical hourly sample."),
    ]
    out = {}
    for key, title, provenance, frame, description in items:
        frames = investigate(frame[["timestamp", *SENSORS]], policy)
        for event, label in zip(frames, frame.label):
            event["ground_truth"] = "Event" if label == 1 else "Normal"
        out[key] = {"id": key, "title": title, "provenance": provenance, "description": description, "cadence_minutes": 60,
                    "policy_id": report["policy_id"], "policy": asdict(policy), "frames": frames, "sensor_labels": {"pressure_bar": "Pressure · bar", "temp_C": "Temperature · °C", "flow_Lps": "Flow · L/s"}}
    return out
