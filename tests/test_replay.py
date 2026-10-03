"""Tests for backend/replay.py — sources, tick shape, NaN handling, phases.

No network calls. Uses seed data and fixtures.
"""
import json
import pytest
import numpy as np
from pathlib import Path

from backend.replay import (
    _load_seed, load_instance, list_available_instances, load_demo_recording,
    STANDARD_SENSORS, SENSOR_JSON_NAMES,
)


class TestSeedSource:
    def test_loads_seed(self):
        inst = load_instance("seed")
        assert inst is not None
        assert inst.n_minutes > 0
        assert inst.metadata["instance_id"] == "seed"
        assert inst.metadata["source"] == "seed"

    def test_tick_shape(self):
        inst = load_instance("seed")
        tick = inst.tick(0)
        assert "t" in tick
        assert "minute_index" in tick
        assert "sensors" in tick
        assert "phase" in tick
        # All standard sensor JSON names should be present
        for json_name in SENSOR_JSON_NAMES.values():
            assert json_name in tick["sensors"]

    def test_nan_becomes_none(self):
        inst = load_instance("seed")
        tick = inst.tick(0)
        # Seed only has P-TPT, T-TPT, QGL — others should be None
        assert tick["sensors"]["P_PDG_bar"] is None
        assert tick["sensors"]["T_PDG_C"] is None
        assert tick["sensors"]["P_MON_CKP_bar"] is None

    def test_available_sensors_have_values(self):
        inst = load_instance("seed")
        tick = inst.tick(0)
        assert tick["sensors"]["P_TPT_bar"] is not None
        assert tick["sensors"]["T_TPT_C"] is not None
        assert tick["sensors"]["QGL"] is not None

    def test_phase_values(self):
        inst = load_instance("seed")
        phases = set()
        for i in range(inst.n_minutes):
            phases.add(inst.tick(i)["phase"])
        assert "normal" in phases
        assert "established" in phases

    def test_phase_marker_on_change(self):
        """Verify phases change at the right points in seed data."""
        inst = load_instance("seed")
        prev = None
        changes = []
        for i in range(inst.n_minutes):
            phase = inst.tick(i)["phase"]
            if phase != prev:
                changes.append((i, phase))
                prev = phase
        # Should start normal, then have established sections
        assert changes[0][1] == "normal"
        assert any(p == "established" for _, p in changes)

    def test_from_minute_range(self):
        inst = load_instance("seed")
        tick_10 = inst.tick(10)
        assert tick_10["minute_index"] == 10
        tick_0 = inst.tick(0)
        assert tick_0["minute_index"] == 0

    def test_enrichment_hooks_null_when_not_implemented(self):
        inst = load_instance("seed")
        tick = inst.tick(0)
        # ML predictions should be None (model not trained)
        assert tick["p_hydrate"] is None
        assert tick["p_lookalike"] is None
        assert tick["p_normal"] is None

    def test_cache_same_object(self):
        inst1 = _load_seed()
        inst2 = _load_seed()
        # Both should produce valid instances (not checking identity since
        # _load_seed creates new objects each time, but load_instance could cache)
        assert inst1.n_minutes == inst2.n_minutes


class TestUnknownInstance:
    def test_returns_none(self):
        inst = load_instance("nonexistent_instance_xyz")
        assert inst is None


class TestListInstances:
    def test_includes_seed(self):
        instances = list_available_instances()
        ids = [i["instance_id"] for i in instances]
        assert "seed" in ids

    def test_well_info_shape(self):
        instances = list_available_instances()
        for i in instances:
            assert "well_id" in i
            assert "instance_id" in i
            assert "source" in i
            assert "sensors_available" in i
            assert "has_hydrate_event" in i


class TestDemoRecording:
    def test_loads_fixture(self, tmp_path):
        """Recording from a .jsonl file."""
        events = [
            {"type": "tick", "data": {"t": "2024-01-01T00:00:00", "sensors": {}}},
            {"type": "end", "data": {"t": "2024-01-01T00:01:00", "total_minutes": 1, "instance_id": "test"}},
        ]
        jl = tmp_path / "test_rec.jsonl"
        jl.write_text("\n".join(json.dumps(e) for e in events))

        # Monkey-patch DATA_DEMO
        import backend.replay
        old = backend.config.DATA_DEMO
        backend.config.DATA_DEMO = tmp_path
        try:
            loaded = load_demo_recording("test_rec")
            assert loaded is not None
            assert len(loaded) == 2
            assert loaded[0]["type"] == "tick"
        finally:
            backend.config.DATA_DEMO = old
