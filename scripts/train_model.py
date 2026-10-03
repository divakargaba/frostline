"""Train and cross-validate the hydrate classifier (src/model.py).

Owner: Data+ML

Steps:
  1. Build causal features for every processed instance (cached in models/).
  2. Freeze FINAL_TEST_WELLS: they take no part in any choice or in training.
  3. Leave-one-well-out over the remaining (development) real wells: train on
     synthetic + other real wells, predict the held-out well at 1-minute resolution.
  4. Score out-of-fold predictions with the CLAUDE.md event definitions at several
     thresholds, with 95% intervals from resampling wells; pick the feature set.
  5. Fit the final model on all development data, save it, and score the frozen
     test wells once. Everything goes to results/model_cv.json.

Usage:
  python scripts/train_model.py                  # both feature sets, LOWO
  python scripts/train_model.py --folds 6        # faster grouped k-fold for iteration
  python scripts/train_model.py --sets relative  # one feature set
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import build_features, feature_columns  # noqa: E402
from src.model import (  # noqa: E402
    FEATURE_SETS, FINE_CLASSES, GROUPS, TRAIN_STEP, fine_label, predict, save_model, train,
)

PROCESSED = Path("data/processed")
CACHE = Path("models/features.parquet")
OUT = Path("results/model_cv.json")
META = ["instance_id", "well_id", "source", "event_class", "phase", "label"]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ALARM_RUN = 3  # minutes at or above threshold before an alarm is raised
HEADLINE_TH = 0.5
N_BOOT = 1000

# Frozen final test set, never used for training or for any modelling choice.
# Doubles as the demo wells, so the live demo replays wells the model has not seen:
#   WELL-00019  4 production-line hydrates (class 8) + normal files; hydrate demo
#   WELL-00042  5 gas-lift-line hydrates (class 9)
#   WELL-00006  choke scaling look-alike (class 7) + normal files; DISMISS demo
FINAL_TEST_WELLS = ["WELL-00006", "WELL-00019", "WELL-00042"]


def _one(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    f = build_features(df).copy()
    f["label"] = fine_label(f["class"])
    f = f[f["label"].notna()]
    cols = feature_columns(f)
    f[cols] = f[cols].astype("float32")
    return f[META + cols].reset_index()


def feature_table(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    paths = sorted(p for p in PROCESSED.glob("*.parquet"))
    print(f"building features for {len(paths)} instances ...", flush=True)
    parts = Parallel(n_jobs=-2)(delayed(_one)(p) for p in paths)
    table = pd.concat(parts, ignore_index=True)
    table["label"] = table["label"].astype(int)
    CACHE.parent.mkdir(exist_ok=True)
    table.to_parquet(CACHE)
    return table


def train_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Every TRAIN_STEP-th minute of each instance."""
    return df[df.groupby("instance_id").cumcount() % TRAIN_STEP == 0]


def ambiguous_instances(df: pd.DataFrame) -> set[str]:
    """Event files (folder 4/6-9) whose minutes are all labelled normal.

    43 of 57 real class-9 files are like this: filed as a hydrate, but no
    minute carries an event label, so their 'normal' labels can't be trusted.
    They are left out of training and false-alarm scoring and reported apart.
    """
    has_event = df.groupby("instance_id")["label"].apply(lambda s: (s > 0).any())
    event_file = df.groupby("instance_id")["event_class"].first() != 0
    return set(has_event.index[event_file & ~has_event])


def cross_validate(table: pd.DataFrame, features: list[str], folds: int | None) -> pd.DataFrame:
    real = table["source"] == "real"
    wells = sorted(table.loc[real, "well_id"].unique())
    if folds:
        rng = np.random.default_rng(0)
        groups = np.array_split(rng.permutation(wells), folds)
    else:
        groups = [[w] for w in wells]
    sub = train_rows(table[~table["instance_id"].isin(ambiguous_instances(table))])
    oof = []
    for i, held in enumerate(groups, 1):
        t0 = time.time()
        test_mask = real & table["well_id"].isin(held)
        model = train(sub[~(sub["source"].eq("real") & sub["well_id"].isin(held))], features=features)
        pred = predict(table.loc[test_mask, features + META + ["timestamp"]], model)
        oof.append(pred[list(dict.fromkeys(META + ["timestamp"] + list(GROUPS) + [f"p_{c}" for c in FINE_CLASSES]))])
        print(f"  fold {i}/{len(groups)} {','.join(map(str, held))}: {test_mask.sum()} min, {time.time() - t0:.0f}s", flush=True)
    return pd.concat(oof, ignore_index=True)


def alarm_runs(p: np.ndarray, th: float) -> np.ndarray:
    """True on minutes where the score has been >= th for ALARM_RUN+ consecutive minutes."""
    above = p >= th
    run = np.zeros(len(p), int)
    for i, a in enumerate(above):
        run[i] = run[i - 1] + 1 if a and i else int(a)
    return run >= ALARM_RUN


def instance_stats(oof: pd.DataFrame, th: float) -> pd.DataFrame:
    """Per-instance outcome at threshold th (CLAUDE.md definitions).

    kind: hydrate (class 8/9 with a forming phase), hydrate_no_forming, lookalike
    (4/6/7), normal (0) or ambiguous. Alarm onsets during normal minutes are false
    alarms; lead times are measured from the start of the catching alarm episode.
    """
    ambiguous = ambiguous_instances(oof)
    per_minute = pd.Timedelta(minutes=1)
    rows = []
    for iid, g in oof.sort_values(["instance_id", "timestamp"]).groupby("instance_id", sort=False):
        on = alarm_runs(g["p_hydrate"].to_numpy(), th)
        scored = g["p_hydrate"].notna().to_numpy()  # warm-up minutes are not scored
        ec = int(g["event_class"].iloc[0])
        r = {"instance_id": iid, "well_id": g["well_id"].iloc[0], "event_class": ec, "outcome": None,
             "lead_est": np.nan, "lead_form": np.nan, "flagged": False, "alarm_min": int(on.sum()),
             "minutes": int(scored.sum())}
        if iid in ambiguous:
            rows.append({**r, "kind": "ambiguous", "fa": 0, "normal_min": 0})
            continue
        phase = g["phase"].to_numpy()
        normal = phase == "normal"
        onset = on & ~np.r_[False, on[:-1]]
        r.update(fa=int((onset & normal).sum()), normal_min=int((normal & scored).sum()))
        if ec in (8, 9):
            forming, est = phase == "forming", phase == "established"
            hit = np.flatnonzero(on & forming)
            if not forming.any():
                r.update(kind="hydrate_no_forming", outcome="late" if (on & est).any() else "missed")
            elif len(hit):
                t = pd.DatetimeIndex(g["timestamp"])
                start = hit[0]
                while start > 0 and on[start - 1]:  # back to the start of that alarm episode
                    start -= 1
                r.update(kind="hydrate", outcome="caught", lead_form=(t[forming.argmax()] - t[start]) / per_minute,
                         lead_est=(t[est.argmax()] - t[start]) / per_minute if est.any() else np.nan)
            else:
                r.update(kind="hydrate", outcome="late" if (on & est).any() else "missed")
        elif ec in (4, 6, 7):
            r.update(kind="lookalike", flagged=bool((on & ~normal).any()))
        else:
            r.update(kind="normal")
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(st: pd.DataFrame, th: float) -> dict:
    """Aggregate instance_stats into the headline event metrics."""
    ev = st[st["kind"].isin(["hydrate", "hydrate_no_forming"])]
    look, amb = st[st["kind"] == "lookalike"], st[st["kind"] == "ambiguous"]
    per_day = lambda d: round(d["fa"].sum() / max(d["normal_min"].sum(), 1) * 1440, 3)  # noqa: E731
    counted = st[st["kind"] != "ambiguous"]
    caught = int((ev["outcome"] == "caught").sum())
    n_ev = len(ev)
    return {
        "threshold": th,
        "caught": caught, "late": int((ev["outcome"] == "late").sum()), "missed": int((ev["outcome"] == "missed").sum()),
        "events": n_ev,
        "caught_rate": round(caught / n_ev, 3) if n_ev else None,
        "false_alarms_per_day": per_day(counted),
        "false_alarms_per_day_normal_files": per_day(counted[counted["kind"] == "normal"]),
        "false_alarms_per_day_event_lead_in": per_day(counted[counted["kind"] != "normal"]),
        "mean_lead_time_min": round(float(ev["lead_est"].mean()), 1) if ev["lead_est"].notna().any() else None,
        "median_lead_vs_form_min": round(float(ev["lead_form"].median()), 1) if ev["lead_form"].notna().any() else None,
        "misdiagnosis_rate": round(float(look["flagged"].mean()), 3) if len(look) else None,
        "lookalike_events": len(look),
        "ambiguous_files_pct_time_in_alarm": round(100 * amb["alarm_min"].sum() / max(amb["minutes"].sum(), 1), 1),
    }


def event_metrics(oof: pd.DataFrame, th: float) -> dict:
    return summarize(instance_stats(oof, th), th)


def bootstrap(st: pd.DataFrame, th: float, n: int = N_BOOT) -> dict:
    """95% intervals from resampling wells with replacement (wells, not minutes, are independent)."""
    rng = np.random.default_rng(0)
    by_well = {w: g for w, g in st.groupby("well_id")}
    wells = np.array(list(by_well))
    keys = ["caught_rate", "false_alarms_per_day", "false_alarms_per_day_normal_files",
            "misdiagnosis_rate", "mean_lead_time_min"]
    draws = {k: [] for k in keys}
    for _ in range(n):
        s = summarize(pd.concat([by_well[w] for w in rng.choice(wells, len(wells))]), th)
        for k in keys:
            if s[k] is not None:
                draws[k].append(s[k])
    return {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None
            for k, v in draws.items()}


def minute_metrics(oof: pd.DataFrame) -> dict:
    oof = oof[oof["p_hydrate"].notna()]
    y = oof["label"].to_numpy()
    pred = oof[[f"p_{c}" for c in FINE_CLASSES]].to_numpy().argmax(1)
    is_h = np.isin(y, [1, 2])
    return {
        "macro_f1": round(f1_score(y, pred, average="macro"), 3),
        "auc_hydrate_vs_rest": round(roc_auc_score(is_h, oof["p_hydrate"]), 3) if 0 < is_h.sum() < len(y) else None,
        "confusion": {"classes": FINE_CLASSES, "matrix": confusion_matrix(y, pred, labels=range(len(FINE_CLASSES))).tolist()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=None, help="grouped k-fold instead of leave-one-well-out")
    ap.add_argument("--sets", nargs="+", default=list(FEATURE_SETS))
    ap.add_argument("--rebuild", action="store_true", help="rebuild the feature cache")
    args = ap.parse_args()

    table = feature_table(args.rebuild)
    frozen = table["source"].eq("real") & table["well_id"].isin(FINAL_TEST_WELLS)
    dev, test = table[~frozen], table[frozen]
    print(f"{len(table)} labelled minutes, {table['instance_id'].nunique()} instances; "
          f"development: {dev.loc[dev.source == 'real', 'well_id'].nunique()} real wells; "
          f"frozen test: {', '.join(FINAL_TEST_WELLS)}", flush=True)
    all_cols = feature_columns(table)

    report = {"split": f"{args.folds}-fold by well" if args.folds else "leave-one-well-out",
              "final_test_wells": FINAL_TEST_WELLS, "development": {}}
    for name in args.sets:
        features = FEATURE_SETS[name](all_cols)
        print(f"\n[{name}] {len(features)} features", flush=True)
        oof = cross_validate(dev, features, args.folds)
        oof.to_parquet(Path("models") / f"oof_{name}.parquet")
        events = [event_metrics(oof, th) for th in THRESHOLDS]
        report["development"][name] = {
            "n_features": len(features),
            "minute": minute_metrics(oof),
            "events": events,
            "ci95_at_headline": bootstrap(instance_stats(oof, HEADLINE_TH), HEADLINE_TH),
        }
        for row in events:
            print("   ", row, flush=True)
        print("    95% CI:", report["development"][name]["ci95_at_headline"], flush=True)

    # Most events caught at the headline threshold, then fewest false alarms (development data only).
    def score(name):
        e = next(r for r in report["development"][name]["events"] if r["threshold"] == HEADLINE_TH)
        return (e["caught"], -e["false_alarms_per_day"])
    best = max(report["development"], key=score)
    report["chosen_feature_set"] = best

    features = FEATURE_SETS[best](all_cols)
    final = train(train_rows(dev[~dev["instance_id"].isin(ambiguous_instances(dev))]), features=features)
    save_model(final)
    imp = pd.Series(final["booster"].feature_importance("gain"), index=final["features"]).sort_values(ascending=False)
    report["top_features"] = {k: round(float(v), 1) for k, v in imp.head(15).items()}

    # Frozen test wells: scored once, by the saved model.
    pred = predict(test[features + META + ["timestamp"]], final)
    pred.to_parquet(Path("models") / "final_test_predictions.parquet")
    st = instance_stats(pred, HEADLINE_TH)
    report["final_test"] = {
        "minute": minute_metrics(pred),
        "events": [event_metrics(pred, th) for th in THRESHOLDS],
        "per_event": st[st["kind"] != "normal"].replace({np.nan: None}).to_dict(orient="records"),
    }
    print("\nFINAL TEST (frozen wells):", report["final_test"]["events"][THRESHOLDS.index(HEADLINE_TH)], flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nchosen: {best}; model saved; report -> {OUT}")


if __name__ == "__main__":
    main()
