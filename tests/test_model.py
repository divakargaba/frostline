"""Tests for src/model.py — LightGBM classifier.

Owner: Data+ML
"""

import numpy as np
import pandas as pd
import pytest

import src.model as model
from src.features import build_features


@pytest.fixture
def tiny_model(fake_3w_1min, monkeypatch):
    """A few-round model trained on two copies of the fake instance (5 minutes long, so no warm-up)."""
    monkeypatch.setattr(model, "NUM_ROUNDS", 5)
    monkeypatch.setattr(model, "WARMUP_MIN", 0)
    monkeypatch.setitem(model.PARAMS, "min_data_in_leaf", 5)
    parts = []
    for i in range(2):
        df = fake_3w_1min.copy()
        df["instance_id"] = f"inst{i}"
        parts.append(build_features(df))
    feats = pd.concat(parts)
    feats["label"] = model.fine_label(feats["class"])
    return model.train(feats[feats["label"].notna()])


class TestLabels:
    def test_fine_label_mapping(self):
        cls = pd.Series([0, 4, 6, 7, 8, 9, 106, 107, 108, 109], dtype="Int64")
        assert model.fine_label(cls).tolist() == [0, 4, 3, 3, 1, 2, 3, 3, 1, 2]

    def test_groups_cover_every_class_once(self):
        members = [m for ms in model.GROUPS.values() for m in ms]
        assert sorted(members) == sorted(model.FINE_CLASSES)


class TestSampleWeights:
    def test_long_and_short_events_weigh_the_same(self):
        df = pd.DataFrame({
            "instance_id": ["a"] * 90 + ["b"] * 10,
            "label": [1] * 100,
            "phase": ["established"] * 100,
            "source": ["real"] * 100,
        })
        w = model.sample_weights(df)
        assert w[:90].sum() == pytest.approx(w[90:].sum())
        assert w.mean() == pytest.approx(1.0)

    def test_forming_gets_extra_weight(self):
        df = pd.DataFrame({
            "instance_id": ["a", "a", "b", "b"],
            "label": [1, 1, 1, 1],
            "phase": ["forming", "forming", "established", "established"],
            "source": ["real"] * 4,
        })
        w = model.sample_weights(df)
        assert w[0] == pytest.approx(model.FORMING_WEIGHT * w[2])


class TestPredictOutput:
    def test_returns_three_probabilities(self, tiny_model, fake_3w_1min):
        """predict() takes processed sensor data and returns p_hydrate, p_lookalike, p_normal summing to 1."""
        result = model.predict(fake_3w_1min, tiny_model)
        probs = result[["p_hydrate", "p_lookalike", "p_normal"]].to_numpy()
        assert ((probs >= 0) & (probs <= 1)).all()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        assert set(result["predicted_class"]) <= set(model.FINE_CLASSES)

    def test_warmup_rows_are_not_scored(self, tiny_model, fake_3w_1min, monkeypatch):
        """No score until the instance has WARMUP_MIN minutes of history."""
        monkeypatch.setattr(model, "WARMUP_MIN", 2)
        result = model.predict(fake_3w_1min, tiny_model)
        assert result["p_hydrate"].iloc[:2].isna().all()
        assert result["predicted_class"].iloc[:2].isna().all()
        assert result["p_hydrate"].iloc[2:].notna().all()

    def test_leaky_baselines_excluded(self):
        """360-min baselines encode recording length (short normal files), so no feature set uses them."""
        cols = ["P-TPT_delta_60", "P-TPT_delta_360", "P-TPT_std_10"]
        for pick in model.FEATURE_SETS.values():
            assert "P-TPT_delta_360" not in pick(cols)

    def test_causal(self, tiny_model, fake_3w_1min):
        """Adding later minutes must not change earlier scores."""
        n = len(fake_3w_1min) // 2
        early = model.predict(fake_3w_1min.iloc[:n], tiny_model)["p_hydrate"].to_numpy()
        full = model.predict(fake_3w_1min, tiny_model)["p_hydrate"].to_numpy()[:n]
        np.testing.assert_allclose(early, full, atol=1e-9)


class TestSaveLoad:
    def test_round_trip(self, tiny_model, fake_3w_1min, tmp_path):
        path = tmp_path / "m.pkl"
        model.save_model(tiny_model, str(path))
        loaded = model.load_model(str(path))
        a = model.predict(fake_3w_1min, tiny_model)["p_hydrate"]
        b = model.predict(fake_3w_1min, loaded)["p_hydrate"]
        np.testing.assert_allclose(a, b)
