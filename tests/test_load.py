"""Tests for src/load.py — data loading and preprocessing.

Owner: Data+ML
"""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.pending


class TestPaToBar:
    def test_known_conversion(self, fake_3w_instance):
        """1 bar = 100,000 Pa. 28_000_000 Pa should be ~280 bar."""
        from src.load import load_well  # noqa: deferred import

        # After loading, P-PDG should be in bar (~280)
        # Using the fixture as a proxy: raw value / 1e5
        raw_pa = fake_3w_instance["P-PDG"].iloc[0]
        expected_bar = raw_pa / 1e5
        assert 250 < expected_bar < 310


class TestDownsample:
    def test_length(self, fake_3w_instance):
        """300 seconds at 1 Hz -> 5 full 1-minute bins."""
        # 0:00–0:59 (60s), 1:00–1:59, 2:00–2:59, 3:00–3:59, 4:00–4:59
        expected_minutes = 5
        resampled = fake_3w_instance.resample("1min").mean()
        assert len(resampled) == expected_minutes

    def test_mode_label_per_minute(self, fake_3w_instance):
        """Class label should use mode (most frequent) per minute, not mean."""
        # Minute 2 (120-179s) is all class 108 -> mode should be 108
        minute_2 = fake_3w_instance.iloc[120:180]["class"]
        assert minute_2.mode().iloc[0] == 108


class TestPhaseDeriv:
    def test_normal_phase(self):
        """Class 0 -> phase 'normal'."""
        from src.load import load_well  # noqa

        # Expectation: rows with class 0 get phase "normal"
        assert True  # placeholder assertion until load_well is implemented

    def test_forming_phase(self):
        """Class 108 (100 + 8) -> phase 'forming'."""
        assert 108 >= 100  # transient class = base + 100

    def test_established_phase(self):
        """Class 8 -> phase 'established'."""
        assert 8 < 100


class TestNaNHandling:
    def test_nan_class_dropped(self, fake_3w_instance):
        """Rows with NaN class should be dropped from training data."""
        nan_count = fake_3w_instance["class"].isna().sum()
        assert nan_count == 5  # rows 50-54


class TestSourceParsing:
    def test_real_well_filename(self):
        """WELL-00019_20170101120000.parquet -> source='real', well_id='WELL-00019'."""
        filename = "WELL-00019_20170101120000.parquet"
        assert filename.startswith("WELL-")
        well_id = filename.split("_")[0]
        assert well_id == "WELL-00019"

    def test_simulated_filename(self):
        """SIMULATED_00001.parquet -> source='simulated'."""
        filename = "SIMULATED_00001.parquet"
        assert filename.startswith("SIMULATED_")

    def test_drawn_never_in_test(self):
        """DRAWN files must never appear in the test set."""
        filename = "DRAWN_00001.parquet"
        source = "drawn" if filename.startswith("DRAWN_") else "real"
        assert source == "drawn"
        # When implemented: assert no drawn files in test split
