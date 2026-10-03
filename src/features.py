"""Feature engineering for hydrate detection.

Owner: Data+ML

Computes causal (backward-looking) rolling features over sensor columns:
  - Rolling mean, std, and slope over 10, 30, and 60-minute windows
  - Baseline deltas (current value minus rolling mean)
  - Pressure differentials (e.g., PDG minus TPT)

All features are strictly causal — no future data leakage.

TODO:
  - Implement rolling statistics (mean, std, slope)
  - Implement baseline delta features
  - Implement pressure differential features
  - Define final feature column list
"""

import pandas as pd


def rolling_stats(df: pd.DataFrame, windows: list[int] = [10, 30, 60]) -> pd.DataFrame:
    """Compute causal rolling mean, std, and slope for sensor columns.

    Args:
        df: Input dataframe with 1-min sensor readings.
        windows: List of window sizes in minutes.

    Returns:
        DataFrame with added rolling feature columns.
    """
    raise NotImplementedError


def baseline_deltas(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Compute delta between current value and rolling baseline mean.

    Args:
        df: Input dataframe with sensor readings.
        window: Baseline window size in minutes.

    Returns:
        DataFrame with added delta columns.
    """
    raise NotImplementedError


def pressure_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pressure differential features (e.g., PDG - TPT).

    Args:
        df: Input dataframe with pressure columns.

    Returns:
        DataFrame with added differential columns.
    """
    raise NotImplementedError


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature-engineering pipeline.

    Args:
        df: Raw preprocessed dataframe from load module.

    Returns:
        Feature-enriched DataFrame ready for modelling.
    """
    raise NotImplementedError
