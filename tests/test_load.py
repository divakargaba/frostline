"""Tests for src/load.py — data loading and preprocessing.

Owner: Data+ML
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.load import (
    SENSORS,
    load_all,
    load_well,
    make_instance_id,
    parse_source,
    phase_from_class,
)

INSTANCE = "WELL-00099_20240101000000"


def _write_raw(df: pd.DataFrame, root: Path, event_class: int, name: str) -> Path:
    """Write a fake raw 3W file to root/<class>/<name>.parquet."""
    folder = root / str(event_class)
    folder.mkdir(parents=True, exist_ok=True)
    raw = df.copy()
    raw["class"] = raw["class"].astype("Int16")
    raw["state"] = pd.array(np.zeros(len(raw)), dtype="Int16")
    path = folder / f"{name}.parquet"
    raw.to_parquet(path)
    return path


@pytest.fixture
def loaded(fake_3w_instance, tmp_path) -> pd.DataFrame:
    return load_well(_write_raw(fake_3w_instance, tmp_path, 8, INSTANCE))


class TestPaToBar:
    def test_known_conversion(self, fake_3w_instance, loaded):
        """1 bar = 100,000 Pa. ~28_000_000 Pa should be ~280 bar."""
        expected = fake_3w_instance["P-PDG"].iloc[:60].mean() / 1e5
        assert loaded["P-PDG"].iloc[0] == pytest.approx(expected)
        assert 250 < loaded["P-PDG"].iloc[0] < 310

    def test_temperature_unchanged(self, fake_3w_instance, loaded):
        expected = fake_3w_instance["T-TPT"].iloc[:60].mean()
        assert loaded["T-TPT"].iloc[0] == pytest.approx(expected)


class TestDownsample:
    def test_length(self, loaded):
        """300 seconds at 1 Hz -> 5 full 1-minute bins."""
        assert len(loaded) == 5
        assert (loaded.index.to_series().diff().dropna() == pd.Timedelta("1min")).all()

    def test_mode_label_per_minute(self, loaded):
        """Class label uses the per-minute mode, not the mean."""
        assert loaded["class"].tolist() == [0, 0, 108, 8, 0]

    def test_label_dtype(self, loaded):
        assert str(loaded["class"].dtype) == "Int64"


class TestPhaseDeriv:
    def test_phases_from_loader(self, loaded):
        assert loaded["phase"].tolist() == [
            "normal", "normal", "forming", "established", "normal"]

    def test_phase_from_class(self):
        cls = pd.Series([0, 108, 109, 8, 9, 4, np.nan])
        assert phase_from_class(cls).tolist() == [
            "normal", "forming", "forming", "established", "established",
            "established", None]


class TestNaNHandling:
    def test_nan_class_dropped(self, fake_3w_instance, tmp_path):
        """Minutes whose class is entirely NaN are dropped."""
        df = fake_3w_instance.copy()
        df.iloc[60:120, df.columns.get_loc("class")] = np.nan  # minute 1
        out = load_well(_write_raw(df, tmp_path, 8, INSTANCE))
        assert len(out) == 4
        assert out["class"].notna().all()
        assert pd.Timestamp("2024-01-01 00:01") not in out.index

    def test_partial_nan_minute_kept(self, loaded):
        """Rows 50-54 are NaN but minute 0 still has a valid mode."""
        assert loaded["class"].iloc[0] == 0

    def test_missing_sensor_flags(self, loaded):
        """T-JUS-CKP is all NaN; absent columns (ABER-CKP) become NaN too."""
        assert not loaded["has_T-JUS-CKP"].any()
        assert not loaded["has_ABER-CKP"].any()
        assert loaded["ABER-CKP"].isna().all()
        assert loaded["has_P-PDG"].all()
        for s in SENSORS:
            assert s in loaded.columns and f"has_{s}" in loaded.columns

    def test_frozen_sensor_treated_as_missing(self, fake_3w_instance, tmp_path):
        """A sensor stuck at one value for the whole file is dead -> NaN."""
        df = fake_3w_instance.copy()
        df["P-PDG"] = 0.0
        df["QGL"] = 12.0
        out = load_well(_write_raw(df, tmp_path, 0, INSTANCE))
        assert out["P-PDG"].isna().all() and not out["has_P-PDG"].any()
        assert out["QGL"].isna().all()
        assert out["P-TPT"].notna().all()  # varying sensors untouched
        assert out.attrs["frozen_sensors"] == ["P-PDG", "QGL"]


class TestSourceParsing:
    def test_real_well_filename(self, loaded):
        assert parse_source("WELL-00019_20170101120000.parquet") == ("real", "WELL-00019")
        assert (loaded["well_id"] == "WELL-00099").all()
        assert (loaded["source"] == "real").all()
        assert (loaded["instance_id"] == INSTANCE).all()
        assert (loaded["event_class"] == 8).all()

    def test_simulated_filename(self):
        assert parse_source("SIMULATED_00001.parquet") == ("simulated", None)

    def test_drawn_filename(self):
        assert parse_source("DRAWN_00001.parquet") == ("drawn", None)

    def test_unknown_filename_rejected(self):
        with pytest.raises(ValueError):
            parse_source("foo.parquet")

    def test_instance_ids_unique_across_folders(self):
        """SIMULATED_ stems repeat across class folders in 3W."""
        assert make_instance_id(Path("8/SIMULATED_00001.parquet")) == "SIMULATED_00001_c8"
        assert make_instance_id(Path("9/SIMULATED_00001.parquet")) == "SIMULATED_00001_c9"
        assert make_instance_id(Path("8/" + INSTANCE + ".parquet")) == INSTANCE


class TestBatch:
    def test_load_all_writes_outputs_and_index(self, fake_3w_instance, tmp_path):
        raw, out = tmp_path / "raw", tmp_path / "processed"
        _write_raw(fake_3w_instance, raw, 8, INSTANCE)
        _write_raw(fake_3w_instance, raw, 8, "SIMULATED_00001")
        _write_raw(fake_3w_instance, raw, 9, "SIMULATED_00001")
        (raw / "7").mkdir()
        (raw / "7" / "WELL-00001_bad.parquet").write_text("not a parquet")

        index = load_all(str(raw), str(out), workers=1)

        assert sorted(index["instance_id"]) == sorted(
            [INSTANCE, "SIMULATED_00001_c8", "SIMULATED_00001_c9"])
        assert (out / f"{INSTANCE}.parquet").exists()
        assert (out / "index.csv").exists()
        row = index.set_index("instance_id").loc[INSTANCE]
        assert row["n_minutes"] == 5
        assert (row["minutes_normal"], row["minutes_forming"], row["minutes_established"]) == (3, 1, 1)
        assert bool(row["has_T-TPT"]) and not bool(row["has_T-JUS-CKP"])

    def test_skip_existing_reuses_output(self, fake_3w_instance, tmp_path):
        raw, out = tmp_path / "raw", tmp_path / "processed"
        df = fake_3w_instance.copy()
        df["QGL"] = 0.0
        _write_raw(df, raw, 8, INSTANCE)
        load_all(str(raw), str(out), workers=1)
        mtime = (out / f"{INSTANCE}.parquet").stat().st_mtime_ns
        index = load_all(str(raw), str(out), workers=1, skip_existing=True)
        assert (out / f"{INSTANCE}.parquet").stat().st_mtime_ns == mtime
        assert len(index) == 1 and index["n_minutes"].iloc[0] == 5
        assert index["frozen_sensors"].iloc[0] == "QGL"
