"""Feature engineering for hydrate detection.

Owner: Data+ML

Input: the 1-minute frame written by src/load.py (one or more instances,
DatetimeIndex "timestamp", pressures in bar). Computes, per instance:
  - Pressure differentials: P-TPT minus P-MON-CKP (line dP), P-PDG minus P-TPT
  - Physics hook: subcooling margin_C from src.physics.hydrate_margin
    (P-TPT, T-TPT); NaN until physics.py is implemented or if a sensor is missing
  - Rolling mean, std, and slope (per minute) over 10, 30, 60-minute windows
  - Baseline deltas: current value minus the 60 and 360-minute rolling mean

All features are strictly causal: every window is time-based and ends at the
current minute, so a row only sees its own and earlier minutes of the same
instance. Time-based windows also handle the gaps left by dropped NaN-class
minutes. Missing sensors stay NaN (LightGBM handles NaN natively).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.load import SENSORS

WINDOWS = [10, 30, 60]
BASELINE_WINDOWS = [60, 360]
DIFFERENTIALS = {
    "P_TPT_minus_P_MON": ("P-TPT", "P-MON-CKP"),
    "P_PDG_minus_P_TPT": ("P-PDG", "P-TPT"),
}
MARGIN = "margin_C"
BASE_COLUMNS = SENSORS + list(DIFFERENTIALS) + [MARGIN]


def _min_periods(window: int) -> int:
    """Require at least half the window (and 2 points) before emitting a stat."""
    return max(2, window // 2)


def pressure_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Add pressure differential columns (NaN if either sensor is missing)."""
    out = df.copy()
    for name, (a, b) in DIFFERENTIALS.items():
        out[name] = out[a] - out[b]
    return out


def _default_margin_fn() -> Callable[[float, float], float] | None:
    """src.physics.hydrate_margin if it is implemented, else None."""
    try:
        from src.physics import hydrate_margin
        hydrate_margin(200.0, 50.0)
    except (ImportError, NotImplementedError):
        return None
    return hydrate_margin


def physics_features(df: pd.DataFrame,
                     margin_fn: Callable[[float, float], float] | None = None) -> pd.DataFrame:
    """Physics hook: add margin_C = margin_fn(P-TPT bar, T-TPT C).

    margin_fn defaults to src.physics.hydrate_margin; while that is not
    implemented, margin_C is all NaN so the feature list never changes.
    """
    out = df.copy()
    fn = margin_fn or _default_margin_fn()
    p, t = out["P-TPT"].to_numpy(float), out["T-TPT"].to_numpy(float)
    margin = np.full(len(out), np.nan)
    if fn is not None:
        ok = ~(np.isnan(p) | np.isnan(t))
        margin[ok] = [fn(pi, ti) for pi, ti in zip(p[ok], t[ok])]
    out[MARGIN] = margin
    return out


def _rolling_slope(x: pd.Series, window: str, min_periods: int) -> pd.Series:
    """Least-squares slope of x vs time (units per minute) over a causal window."""
    t = pd.Series((x.index - x.index[0]) / pd.Timedelta("1min"), index=x.index)
    t = t.where(x.notna())
    roll = lambda s: s.rolling(window, min_periods=min_periods).mean()  # noqa: E731
    mean_t, mean_x = roll(t), roll(x)
    cov = roll(t * x) - mean_t * mean_x
    var = roll(t * t) - mean_t ** 2
    return cov / var.where(var > 1e-9)


def rolling_stats(df: pd.DataFrame, windows: list[int] = WINDOWS,
                  columns: list[str] | None = None) -> pd.DataFrame:
    """Add causal rolling <col>_mean_<w>, <col>_std_<w>, <col>_slope_<w>."""
    out = df.copy()
    cols = [c for c in (columns or BASE_COLUMNS) if c in out.columns]
    new = {}
    for w in windows:
        win, mp = f"{w}min", _min_periods(w)
        for c in cols:
            r = out[c].rolling(win, min_periods=mp)
            new[f"{c}_mean_{w}"] = r.mean()
            new[f"{c}_std_{w}"] = r.std()
            new[f"{c}_slope_{w}"] = _rolling_slope(out[c], win, mp)
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def baseline_deltas(df: pd.DataFrame, window: int = 60,
                    columns: list[str] | None = None) -> pd.DataFrame:
    """Add <col>_delta_<window>: current value minus causal rolling mean."""
    out = df.copy()
    cols = [c for c in (columns or BASE_COLUMNS) if c in out.columns]
    base = out[cols].rolling(f"{window}min", min_periods=_min_periods(window)).mean()
    deltas = (out[cols] - base).add_suffix(f"_delta_{window}")
    return pd.concat([out, deltas], axis=1)


def _build_one(df: pd.DataFrame, margin_fn) -> pd.DataFrame:
    df = df.sort_index()
    for s in SENSORS:
        if s not in df.columns:
            df[s] = np.nan
    df = pressure_differentials(df)
    df = physics_features(df, margin_fn)
    df = rolling_stats(df)
    for w in BASELINE_WINDOWS:
        df = baseline_deltas(df, w)
    return df


def build_features(df: pd.DataFrame,
                   margin_fn: Callable[[float, float], float] | None = None) -> pd.DataFrame:
    """Full feature pipeline. Rolling windows never cross instance boundaries.

    Args:
        df: Processed frame from src.load (one or many instances).
        margin_fn: Optional override for the physics margin (p_bar, t_c) -> C.
    """
    if "instance_id" not in df.columns or df["instance_id"].nunique() <= 1:
        return _build_one(df, margin_fn)
    parts = [_build_one(g, margin_fn) for _, g in df.groupby("instance_id", sort=False)]
    return pd.concat(parts)


def feature_columns(df: pd.DataFrame | None = None) -> list[str]:
    """Model input columns. Excludes labels, ids and has_<sensor> flags, which
    describe the recording setup rather than the well and would leak the label."""
    cols = list(BASE_COLUMNS)
    for w in WINDOWS:
        for c in BASE_COLUMNS:
            cols += [f"{c}_mean_{w}", f"{c}_std_{w}", f"{c}_slope_{w}"]
    for w in BASELINE_WINDOWS:
        cols += [f"{c}_delta_{w}" for c in BASE_COLUMNS]
    if df is not None:
        cols = [c for c in cols if c in df.columns]
    return cols
