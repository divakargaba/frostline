"""LightGBM event classifier.

Owner: Data+ML

Answers, for every minute of a well using only its past: is this a hydrate,
a hydrate look-alike, or normal operation?

The model trains on five fine classes and sums them into the three
probabilities the watcher, tools and API use:

  normal               3W class 0                  -> p_normal
  hydrate_production   3W class 8 (+108 forming)   -> p_hydrate
  hydrate_service      3W class 9 (+109 forming)   -> p_hydrate
  choke_restriction    3W class 6, 7 (+106, 107)   -> p_lookalike
  flow_instability     3W class 4                  -> p_lookalike

Hydrate = forming + established minutes. Early warning is judged at event
level (alarm before the established phase), not per minute.

Training rules (CLAUDE.md):
  - Split by well, never by row. Real wells are held out one at a time;
    SIMULATED / DRAWN instances have no well and only ever train.
  - Each event carries equal total weight, so one long event can't dominate,
    then classes are balanced. Forming minutes get extra weight.
  - Rows are subsampled every TRAIN_STEP minutes (neighbours are near-duplicates).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import build_features, feature_columns

MODEL_PATH = "models/lgbm_hydrate.pkl"

FINE_CLASSES = ["normal", "hydrate_production", "hydrate_service", "choke_restriction", "flow_instability"]
CLASS_OF_3W = {0: 0, 8: 1, 9: 2, 6: 3, 7: 3, 4: 4}
GROUPS = {
    "p_normal": ["normal"],
    "p_hydrate": ["hydrate_production", "hydrate_service"],
    "p_lookalike": ["choke_restriction", "flow_instability"],
}

TRAIN_STEP = 5
# Causal exponential smoothing of the per-minute probabilities (span in minutes).
# Cuts alarm flicker: in 6-fold CV, false alarms/day 2.9 -> 1.4 with the same events caught.
SMOOTH_SPAN = 10
FORMING_WEIGHT = 2.0
SYNTHETIC_WEIGHT = 0.5

PARAMS = {
    "objective": "multiclass",
    "num_class": len(FINE_CLASSES),
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}
# Held-out AUC plateaus by ~150 rounds (0.90 / 0.93 / 0.93 / 0.93 at 50 / 150 / 300 / 600)
# while caught events fall past it (27 / 27 / 26 / 24): more rounds only memorise.
NUM_ROUNDS = 150

# Recording-length leak: 3W normal files are ~300 min long and flow-instability files
# ~120 min, while hydrate and scaling files run for days. Any feature that is still
# empty early in a recording therefore tells the model "short file = normal".
# So: no 360-min baselines (empty for most of every normal file), and no training,
# scoring or prediction in the first WARMUP_MIN minutes, by which point every
# remaining window (<= 60 min) is filled.
WARMUP_MIN = 60
LEAKY = ("_delta_360",)

# Feature sets compared in cross-validation. "relative" also drops absolute levels
# (raw values and rolling means), which differ by well and invite memorising wells.
FEATURE_SETS = {
    "all": lambda cols: [c for c in cols if not c.endswith(LEAKY)],
    "relative": lambda cols: [c for c in cols if any(k in c for k in ("_std_", "_slope_", "_delta_"))
                              and not c.endswith(LEAKY)],
}


def fine_label(cls: pd.Series) -> pd.Series:
    """3W per-minute class (0, 4, 6-9, 106-109) -> fine class index."""
    return (cls.astype("Int64") % 100).map(CLASS_OF_3W).astype("Int64")


def minutes_since_start(df: pd.DataFrame) -> np.ndarray:
    """Minutes since each instance's first row (timestamp column or DatetimeIndex)."""
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.Series(df.index, index=df.index)
    keys = df["instance_id"] if "instance_id" in df.columns else pd.Series(0, index=df.index)
    first = ts.groupby(keys.to_numpy()).transform("min")
    return ((ts - first) / pd.Timedelta(minutes=1)).to_numpy()


def sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Equal total weight per (instance, label), balanced classes, extra weight on forming."""
    key = df["instance_id"].astype(str) + "|" + df["label"].astype(str)
    w = 1.0 / key.map(key.value_counts()).to_numpy(float)
    per_class = pd.Series(w).groupby(df["label"].to_numpy()).sum()
    w = w / df["label"].map(per_class).to_numpy(float)
    w = np.where(df["phase"].to_numpy() == "forming", w * FORMING_WEIGHT, w)
    w = np.where(df["source"].to_numpy() != "real", w * SYNTHETIC_WEIGHT, w)
    return w / w.mean()


def train(feature_df: pd.DataFrame, target_col: str = "label", features: list[str] | None = None) -> dict:
    """Fit the classifier on every row given (call per fold for cross-validation).

    Args:
        feature_df: Output of build_features with `label`, `instance_id`, `phase`, `source`.
        target_col: Fine-class label column (see fine_label).
        features: Feature columns; defaults to the "relative" set.

    Returns:
        Model bundle: {"booster", "features", "classes"}.
    """
    features = features or FEATURE_SETS["relative"](feature_columns(feature_df))
    df = feature_df[feature_df[target_col].notna() & (minutes_since_start(feature_df) >= WARMUP_MIN)]
    ds = lgb.Dataset(df[features].astype("float32"), label=df[target_col].astype(int),
                     weight=sample_weights(df.rename(columns={target_col: "label"})), free_raw_data=True)
    booster = lgb.train(PARAMS, ds, num_boost_round=NUM_ROUNDS)
    return {"booster": booster, "features": features, "classes": FINE_CLASSES}


_cached: dict | None = None


def predict(window_df: pd.DataFrame, model: dict | None = None) -> pd.DataFrame:
    """Score every row of a window.

    Accepts either processed sensor data (features are built here) or an
    already feature-enriched frame. For live use, pass the whole history up to
    now: rolling windows reach back up to 60 minutes.

    Returns:
        Copy of the input with p_normal, p_hydrate, p_lookalike (summing to 1),
        one p_<fine class> column per fine class, and predicted_class. The first
        WARMUP_MIN minutes of each instance are NaN / None: not enough history yet.
    """
    global _cached
    if model is None:
        _cached = _cached or load_model(MODEL_PATH)
        model = _cached
    feats = model["features"]
    df = window_df if all(c in window_df.columns for c in feats) else build_features(window_df)
    out = window_df.copy() if df is window_df else df.copy()
    probs = pd.DataFrame(model["booster"].predict(df[feats].astype("float32")), index=out.index)
    probs[minutes_since_start(out) < WARMUP_MIN] = np.nan
    if SMOOTH_SPAN > 1:  # rows must be in time order within each instance
        keys = out["instance_id"] if "instance_id" in out.columns else pd.Series(0, index=out.index)
        probs = probs.groupby(keys.to_numpy()).transform(lambda s: s.ewm(span=SMOOTH_SPAN).mean())
    probs = probs.to_numpy()
    for i, name in enumerate(model["classes"]):
        out[f"p_{name}"] = probs[:, i]
    for group, members in GROUPS.items():
        out[group] = out[[f"p_{m}" for m in members]].sum(axis=1, min_count=1)
    ready = ~np.isnan(probs).any(axis=1)
    out["predicted_class"] = np.where(ready, np.array(model["classes"])[np.nan_to_num(probs).argmax(axis=1)], None)
    return out


def save_model(model: dict, path: str = MODEL_PATH) -> None:
    """Save a model bundle with joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = MODEL_PATH) -> dict:
    """Load a model bundle saved by save_model."""
    return joblib.load(path)
