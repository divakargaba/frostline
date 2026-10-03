"""LightGBM event classifier and onset detector.

Owner: Data+ML

Trains a LightGBM classifier to detect hydrate-related events using
leave-one-well-out cross-validation. Supports sim-to-real transfer
(train on simulated, evaluate on real).

TODO:
  - Implement leave-one-well-out CV split
  - Implement LightGBM training with class weights
  - Implement sim-to-real transfer training
  - Implement onset detection (first forming prediction in a sequence)
  - Implement model save/load
"""

import pandas as pd


def train(feature_df: pd.DataFrame, target_col: str = "phase") -> object:
    """Train a LightGBM classifier with leave-one-well-out CV.

    Args:
        feature_df: Feature-enriched dataframe with labels.
        target_col: Name of the target column.

    Returns:
        Trained LightGBM Booster or sklearn-compatible model.
    """
    raise NotImplementedError


def predict(window_df: pd.DataFrame, model: object = None) -> pd.DataFrame:
    """Classify events in a sliding window of sensor data.

    Args:
        window_df: Feature-enriched window dataframe.
        model: Trained model (loaded from disk if None).

    Returns:
        DataFrame with added prediction and probability columns.
    """
    raise NotImplementedError


def save_model(model: object, path: str = "models/lgbm_hydrate.pkl") -> None:
    """Serialize a trained model to disk.

    Args:
        model: Trained model object.
        path: Output file path.
    """
    raise NotImplementedError


def load_model(path: str = "models/lgbm_hydrate.pkl") -> object:
    """Load a trained model from disk.

    Args:
        path: Path to the serialized model.

    Returns:
        Trained model object.
    """
    raise NotImplementedError
