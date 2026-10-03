"""Onset forecasting — time to established hydrate phase.

Owner: Physics+Eval

Provides quantile regression (0.1 / 0.5 / 0.9) for estimating minutes
until the hydrate transitions from forming to established phase, plus a
physics-based extrapolation baseline using subcooling rate of change.

TODO:
  - Implement quantile regression model training
  - Implement physics extrapolation baseline
  - Implement forecast function that returns prediction intervals
"""

import pandas as pd


def train_quantile_model(feature_df: pd.DataFrame) -> object:
    """Train quantile regression models for onset time prediction.

    Fits three models at quantiles 0.1, 0.5, 0.9 to predict minutes
    from first forming detection to established phase.

    Args:
        feature_df: Feature dataframe with onset timing labels.

    Returns:
        Dict mapping quantile -> trained model.
    """
    raise NotImplementedError


def physics_extrapolation(window_df: pd.DataFrame, sg: float = 0.65) -> dict:
    """Estimate time to established phase using subcooling trend extrapolation.

    Args:
        window_df: Recent window of sensor data with pressure and temperature.
        sg: Gas specific gravity.

    Returns:
        Dict with keys: minutes_to_established, subcooling_rate_c_per_min.
    """
    raise NotImplementedError


def forecast_onset(window_df: pd.DataFrame, model: object = None) -> dict:
    """Forecast minutes to established hydrate phase.

    Combines ML quantile regression with physics extrapolation.

    Args:
        window_df: Recent window of feature-enriched sensor data.
        model: Trained quantile model dict (loaded from disk if None).

    Returns:
        Dict with keys: q10, q50, q90, physics_estimate.
    """
    raise NotImplementedError
