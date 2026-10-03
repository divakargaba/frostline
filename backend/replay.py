"""Replay engine — loads well data from processed, raw, or seed sources.

Owner: Div

Source selection order:
  1. ProcessedSource: data/processed/<instance_id>.parquet + index.csv
  2. RawSource: data/raw/3W/dataset/ via src.load.load_well (LRU cached)
  3. SeedSource: data/seed/well_hydrate_seed.csv (hourly synthetic)

Each tick is a dict with: t, minute_index, sensors, phase, plus metadata.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.config import DATA_PROCESSED, DATA_RAW_3W, DATA_SEED

log = logging.getLogger("frostline.replay")


def _predict(df: pd.DataFrame) -> pd.DataFrame:
    from src.model import predict
    return predict(df)

STANDARD_SENSORS = [
    "P-PDG", "T-PDG", "P-TPT", "T-TPT", "P-MON-CKP",
    "P-JUS-CKP", "T-JUS-CKP", "ABER-CKP", "QGL",
]

# Friendly names for JSON output (underscores, units)
SENSOR_JSON_NAMES = {
    "P-PDG": "P_PDG_bar",
    "T-PDG": "T_PDG_C",
    "P-TPT": "P_TPT_bar",
    "T-TPT": "T_TPT_C",
    "P-MON-CKP": "P_MON_CKP_bar",
    "P-JUS-CKP": "P_JUS_CKP_bar",
    "T-JUS-CKP": "T_JUS_CKP_C",
    "ABER-CKP": "ABER_CKP",
    "QGL": "QGL",
}


def _clean_value(v):
    """Convert numpy types to Python, NaN/inf to None."""
    if v is None:
        return None
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def _nan_to_none(d: dict) -> dict:
    """Replace NaN/inf with None, numpy types to Python, for JSON."""
    return {k: _clean_value(v) for k, v in d.items()}


def _safe_call(fn, *args, **kwargs):
    """Call fn; return None on NotImplementedError / ImportError / exception."""
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, float) and np.isnan(result):
            return None
        return result
    except (NotImplementedError, ImportError, Exception) as e:
        log.debug("Enrichment hook failed: %s", e)
        return None


class ReplayInstance:
    """Pre-enriched instance data, ready to iterate ticks."""

    def __init__(self, df: pd.DataFrame, metadata: dict):
        self.df = df
        self.metadata = metadata
        self._enrich()

    def _enrich(self):
        """Add margin_C, p_hydrate/lookalike/normal columns if possible."""
        # Physics: hydrate_margin
        try:
            from src.physics import hydrate_margin
            margins = []
            for _, row in self.df.iterrows():
                p = row.get("P-TPT")
                t = row.get("T-TPT")
                if p is not None and t is not None and not (np.isnan(p) or np.isnan(t)):
                    margins.append(_safe_call(hydrate_margin, float(p), float(t)))
                else:
                    margins.append(None)
            self.df["margin_C"] = margins
        except (ImportError, NotImplementedError):
            self.df["margin_C"] = None

        # ML: predict over the whole history at once (features and smoothing are causal).
        # Skipped for the hourly seed CSV; the model is trained on 1-minute 3W data.
        self.df["p_hydrate"] = None
        self.df["p_lookalike"] = None
        self.df["p_normal"] = None
        if self.metadata.get("source") != "seed":
            pred = _safe_call(_predict, self.df)
            if pred is not None:
                for c in ("p_hydrate", "p_lookalike", "p_normal"):
                    self.df[c] = pred[c].to_numpy()

    @property
    def n_minutes(self) -> int:
        return len(self.df)

    def tick(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sensors = {}
        for s in STANDARD_SENSORS:
            sensors[SENSOR_JSON_NAMES[s]] = _clean_value(row.get(s))

        phase = row.get("phase")
        if phase is None or (isinstance(phase, float) and np.isnan(phase)):
            phase = "normal"

        t = row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp(row.name)

        result = {
            "t": t.isoformat(),
            "minute_index": idx,
            "sensors": sensors,
            "phase": phase,
            "margin_C": _clean_value(row.get("margin_C")),
            "p_hydrate": _clean_value(row.get("p_hydrate")),
            "p_lookalike": _clean_value(row.get("p_lookalike")),
            "p_normal": _clean_value(row.get("p_normal")),
        }
        result.update(self.metadata)
        return result


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _load_processed(instance_id: str) -> ReplayInstance | None:
    """Load from data/processed/<instance_id>.parquet."""
    path = DATA_PROCESSED / f"{instance_id}.parquet"
    index_path = DATA_PROCESSED / "index.csv"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)

    meta = {"instance_id": instance_id, "source": "processed"}
    if index_path.exists():
        idx = pd.read_csv(index_path)
        row = idx[idx["instance_id"] == instance_id]
        if len(row):
            r = row.iloc[0]
            meta["well_id"] = r.get("well_id", "")
            meta["event_class"] = int(r.get("event_class", 0))
            meta["sensors_available"] = [
                s for s in STANDARD_SENSORS if r.get(f"has_{s}", False)
            ]
    return ReplayInstance(df, meta)


@functools.lru_cache(maxsize=10)
def _load_raw_cached(parquet_path: str) -> pd.DataFrame:
    from src.load import load_well
    return load_well(parquet_path)


def _find_raw_file(instance_id: str) -> Path | None:
    """Find a raw 3W parquet by instance_id."""
    if not DATA_RAW_3W.exists():
        return None
    for class_dir in sorted(DATA_RAW_3W.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        for f in class_dir.glob("*.parquet"):
            from src.load import make_instance_id
            if make_instance_id(f) == instance_id:
                return f
    return None


def _load_raw(instance_id: str) -> ReplayInstance | None:
    """Load from raw 3W via src.load.load_well (LRU cached)."""
    raw_path = _find_raw_file(instance_id)
    if raw_path is None:
        return None
    df = _load_raw_cached(str(raw_path))
    from src.load import parse_source
    source, well_id = parse_source(raw_path.name)
    event_class = int(raw_path.parent.name)
    sensors_available = [s for s in STANDARD_SENSORS if df[s].notna().any()]
    meta = {
        "instance_id": instance_id,
        "well_id": well_id,
        "source": source,
        "event_class": event_class,
        "sensors_available": sensors_available,
    }
    return ReplayInstance(df, meta)


def _load_seed() -> ReplayInstance:
    """Load seed CSV. Maps: pressure_bar->P-TPT, temp_C->T-TPT, flow_Lps->QGL."""
    path = DATA_SEED / "well_hydrate_seed.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")

    out = pd.DataFrame(index=df.index)
    for s in STANDARD_SENSORS:
        out[s] = np.nan
    out["P-TPT"] = df["pressure_bar"]
    out["T-TPT"] = df["temp_C"]
    out["QGL"] = df["flow_Lps"]

    # Phase: label 0 = normal, label 1 = established (no forming in seed)
    out["phase"] = df["label"].map({0: "normal", 1: "established"})

    meta = {
        "instance_id": "seed",
        "well_id": "SEED",
        "source": "seed",
        "event_class": 8,
        "sensors_available": ["P-TPT", "T-TPT", "QGL"],
    }
    return ReplayInstance(out, meta)


def load_instance(instance_id: str) -> ReplayInstance | None:
    """Load an instance by id, trying processed -> raw -> seed."""
    if instance_id == "seed":
        return _load_seed()

    # Try processed first
    inst = _load_processed(instance_id)
    if inst:
        log.info("Loaded %s from processed", instance_id)
        return inst

    # Try raw 3W
    inst = _load_raw(instance_id)
    if inst:
        log.info("Loaded %s from raw 3W", instance_id)
        return inst

    return None


# ---------------------------------------------------------------------------
# Demo recordings (cached mode)
# ---------------------------------------------------------------------------

def load_demo_recording(instance_id: str) -> list[dict] | None:
    """Load a pre-recorded .jsonl from data/demo/."""
    import json
    from backend.config import DATA_DEMO
    # Try exact match first, then prefix match
    for p in sorted(DATA_DEMO.glob(f"{instance_id}*.jsonl")):
        events = []
        for line in p.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events
    return None


# ---------------------------------------------------------------------------
# Well listing
# ---------------------------------------------------------------------------

def list_available_instances() -> list[dict]:
    """All instances available for replay (processed + raw real 8/9 + seed + demo)."""
    seen: set[str] = set()
    instances: list[dict] = []

    # Processed
    if DATA_PROCESSED.exists():
        index_path = DATA_PROCESSED / "index.csv"
        if index_path.exists():
            idx = pd.read_csv(index_path)
            for _, r in idx.iterrows():
                iid = r["instance_id"]
                # SIMULATED / DRAWN instances have no well and are training-only (CLAUDE.md).
                if iid in seen or pd.isna(r.get("well_id")):
                    continue
                seen.add(iid)
                has_hydrate = int(r.get("event_class", 0)) in (8, 9)
                has_forming = int(r.get("minutes_forming", 0)) > 0
                instances.append({
                    "well_id": r.get("well_id", ""),
                    "instance_id": iid,
                    "source": r.get("source", "real"),
                    "sensors_available": [
                        s for s in STANDARD_SENSORS if r.get(f"has_{s}", False)
                    ],
                    "has_hydrate_event": has_hydrate,
                    "has_forming_phase": has_forming,
                    "n_minutes": int(r.get("n_minutes", 0)),
                })

    # Raw real class 8 and 9
    if DATA_RAW_3W.exists():
        from src.load import make_instance_id, parse_source
        for cls in (8, 9):
            class_dir = DATA_RAW_3W / str(cls)
            if not class_dir.exists():
                continue
            for f in sorted(class_dir.glob("WELL-*.parquet")):
                iid = make_instance_id(f)
                if iid in seen:
                    continue
                seen.add(iid)
                _, well_id = parse_source(f.name)
                instances.append({
                    "well_id": well_id or "",
                    "instance_id": iid,
                    "source": "real",
                    "sensors_available": [],  # Would need to read file to check
                    "has_hydrate_event": True,
                    "has_forming_phase": True,
                    "n_minutes": 0,
                })

    # Seed
    if (DATA_SEED / "well_hydrate_seed.csv").exists() and "seed" not in seen:
        seen.add("seed")
        instances.append({
            "well_id": "SEED",
            "instance_id": "seed",
            "source": "seed",
            "sensors_available": ["P-TPT", "T-TPT", "QGL"],
            "has_hydrate_event": True,
            "has_forming_phase": False,
            "n_minutes": 721,
        })

    return instances
