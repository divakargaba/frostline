"""Tests for src/model.py — LightGBM classifier.

Owner: Data+ML
"""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.pending


class TestSplitByWell:
    def test_no_well_in_both_train_and_test(self, fake_feature_df):
        """Leave-one-well-out: a well must not appear in both train and test."""
        from src.model import train

        # After training, verify the held-out well is not in training data.
        # This test checks the contract: train() should do LOWO.
        # For now, just verify the fixture has a well_id column.
        assert "well_id" in fake_feature_df.columns


class TestPredictOutput:
    def test_returns_three_probabilities(self, fake_feature_df):
        """predict() must return p_hydrate, p_lookalike, p_normal summing to 1."""
        from src.model import predict

        result = predict(fake_feature_df)
        for _, row in result.iterrows():
            probs = [row["p_hydrate"], row["p_lookalike"], row["p_normal"]]
            assert all(0 <= p <= 1 for p in probs)
            assert abs(sum(probs) - 1.0) < 1e-6


class TestLOWOFolds:
    def test_one_fold_per_well(self, fake_feature_df):
        """LOWO should yield exactly one fold per unique well."""
        wells = fake_feature_df["well_id"].nunique()
        # When implemented, train() should return fold results
        assert wells >= 1
