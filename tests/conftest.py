"""Shared test fixtures — synthetic 3W-style data.

Provides a small, deterministic fake instance with known phases,
missing sensors, and NaN labels for testing all modules.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_3w_instance() -> pd.DataFrame:
    """A synthetic 3W-style instance with ~300 rows at 1-second intervals.

    Layout (in seconds, 1 Hz like real 3W):
      0–119:   normal (class 0)
      120–179: forming (class 108)
      180–239: established (class 8)
      240–299: normal again (class 0)

    Sensors included: P-PDG, T-PDG, P-TPT, T-TPT, P-MON-CKP, QGL, class
    T-JUS-CKP is entirely NaN (simulating a missing sensor).
    A few rows have NaN class (rows 50–54) to test NaN-label dropping.
    Pressures are in Pa (will be converted to bar by loader).
    """
    n = 300
    rng = np.random.RandomState(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="1s")

    # Base pressures in Pa (~280 bar = 28_000_000 Pa)
    p_pdg = 28_000_000 + rng.normal(0, 100_000, n)
    p_tpt = 27_800_000 + rng.normal(0, 100_000, n)
    p_mon = 27_500_000 + rng.normal(0, 100_000, n)

    # During forming/established, pressure drops
    for i in range(120, 240):
        drop = 500_000 * ((i - 120) / 120)
        p_pdg[i] -= drop
        p_tpt[i] -= drop

    t_pdg = 85.0 + rng.normal(0, 1, n)
    t_tpt = 84.0 + rng.normal(0, 1, n)
    # During forming/established, temperature drops
    for i in range(120, 240):
        t_tpt[i] -= 3 * ((i - 120) / 120)

    qgl = 12.0 + rng.normal(0, 0.5, n)

    # Class labels
    cls = np.zeros(n, dtype="float64")
    cls[120:180] = 108  # forming
    cls[180:240] = 8    # established
    # NaN labels for rows 50-54
    cls[50:55] = np.nan

    df = pd.DataFrame({
        "P-PDG": p_pdg,
        "T-PDG": t_pdg,
        "P-TPT": p_tpt,
        "T-TPT": t_tpt,
        "P-MON-CKP": p_mon,
        "T-JUS-CKP": np.full(n, np.nan),  # missing sensor
        "QGL": qgl,
        "class": cls,
    }, index=ts)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def fake_3w_1min(fake_3w_instance: pd.DataFrame) -> pd.DataFrame:
    """The fake instance downsampled to 1-minute resolution (5 rows).

    Pressures already converted to bar. Phase column derived.
    This is what the loader output should look like.
    """
    df = fake_3w_instance.copy()
    # Convert Pa to bar
    pressure_cols = ["P-PDG", "P-TPT", "P-MON-CKP"]
    for col in pressure_cols:
        df[col] = df[col] / 1e5

    # Resample to 1-min
    sensor_cols = ["P-PDG", "T-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "QGL"]
    sensors = df[sensor_cols].resample("1min").mean()
    labels = df[["class"]].resample("1min").apply(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
    )
    result = sensors.join(labels)

    # Derive phase
    def _phase(c):
        if pd.isna(c):
            return None
        c = int(c)
        if c == 0:
            return "normal"
        elif c >= 100:
            return "forming"
        else:
            return "established"

    result["phase"] = result["class"].apply(_phase)
    result["well_id"] = "WELL-00099"
    result["source"] = "real"
    result["instance_id"] = "WELL-00099_20240101000000"
    return result


@pytest.fixture
def fake_feature_df(fake_3w_1min: pd.DataFrame) -> pd.DataFrame:
    """Fake feature-enriched dataframe for model tests.

    Adds dummy feature columns so model tests can check shape/split logic.
    """
    df = fake_3w_1min.copy()
    rng = np.random.RandomState(42)
    for window in [10, 30, 60]:
        for col in ["P-PDG", "P-TPT"]:
            df[f"{col}_mean_{window}"] = df[col]  # dummy: same as raw
            df[f"{col}_std_{window}"] = rng.uniform(0, 1, len(df))
    df["P_TPT_minus_P_MON"] = df["P-TPT"] - df["P-MON-CKP"]
    return df
