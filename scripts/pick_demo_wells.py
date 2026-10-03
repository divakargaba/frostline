#!/usr/bin/env python3
"""Pick the best demo wells for the live demo.

Ranks real hydrate instances (class 8) and scaling instances (class 7)
for the live demo based on: has T-TPT, has forming phase, clear transition,
reasonable length, few missing values.

Usage: python scripts/pick_demo_wells.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

from src.load import (
    SENSORS, discover_parquet_files, load_well, make_instance_id,
    parse_source,
)
from backend.config import DATA_DEMO, DATA_RAW_3W


def score_hydrate(path: Path) -> dict | None:
    """Score a real class 8 instance for demo suitability."""
    source, well_id = parse_source(path.name)
    if source != "real":
        return None
    iid = make_instance_id(path)

    schema_names = set(pq.read_schema(path).names)
    has_t_tpt = "T-TPT" in schema_names

    try:
        df = load_well(path)
    except Exception as e:
        return None

    n = len(df)
    if n < 60:
        return None

    phases = df["phase"].value_counts()
    n_forming = int(phases.get("forming", 0))
    n_established = int(phases.get("established", 0))
    n_normal = int(phases.get("normal", 0))

    # Missing values fraction across key sensors
    key_sensors = ["P-PDG", "T-PDG", "P-TPT"]
    if has_t_tpt:
        key_sensors.append("T-TPT")
    missing_frac = df[key_sensors].isna().mean().mean()

    # Score: higher is better
    score = 0
    if has_t_tpt:
        score += 30  # Needed for physics margin
    if n_forming > 0:
        score += 25
    if n_established > 0:
        score += 10
    if n_forming > 30:
        score += 10  # Long enough forming phase to show alert
    score += max(0, 20 - int(missing_frac * 100))  # Low missing is good
    if 200 < n < 20000:
        score += 5  # Reasonable length for demo

    reasons = []
    if has_t_tpt:
        reasons.append("has T-TPT")
    if n_forming > 0:
        reasons.append(f"{n_forming} min forming")
    if n_established > 0:
        reasons.append(f"{n_established} min established")
    reasons.append(f"{missing_frac:.1%} missing")
    reasons.append(f"{n} total min")

    return {
        "instance_id": iid,
        "well_id": well_id,
        "score": score,
        "n_minutes": n,
        "n_forming": n_forming,
        "n_established": n_established,
        "has_T_TPT": has_t_tpt,
        "missing_frac": round(missing_frac, 3),
        "reasons": reasons,
    }


def score_scaling(path: Path) -> dict | None:
    """Score a real class 7 instance for DISMISS demo."""
    source, well_id = parse_source(path.name)
    if source != "real":
        return None
    iid = make_instance_id(path)

    try:
        df = load_well(path)
    except Exception:
        return None

    n = len(df)
    if n < 30:
        return None

    schema_names = set(pq.read_schema(path).names)
    has_t_tpt = "T-TPT" in schema_names
    missing_frac = df[["P-PDG", "T-PDG", "P-TPT"]].isna().mean().mean()

    score = 0
    if has_t_tpt:
        score += 20
    score += max(0, 20 - int(missing_frac * 100))
    if 100 < n < 10000:
        score += 10

    reasons = []
    if has_t_tpt:
        reasons.append("has T-TPT")
    reasons.append(f"{missing_frac:.1%} missing")
    reasons.append(f"{n} total min")

    return {
        "instance_id": iid,
        "well_id": well_id,
        "score": score,
        "n_minutes": n,
        "has_T_TPT": has_t_tpt,
        "missing_frac": round(missing_frac, 3),
        "reasons": reasons,
    }


def main():
    # Class 8 hydrate
    class_8_dir = DATA_RAW_3W / "8"
    hydrate_candidates = []
    if class_8_dir.exists():
        for f in sorted(class_8_dir.glob("WELL-*.parquet")):
            r = score_hydrate(f)
            if r:
                hydrate_candidates.append(r)
    hydrate_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Class 7 scaling
    class_7_dir = DATA_RAW_3W / "7"
    scaling_candidates = []
    if class_7_dir.exists():
        for f in sorted(class_7_dir.glob("WELL-*.parquet")):
            r = score_scaling(f)
            if r:
                scaling_candidates.append(r)
    scaling_candidates.sort(key=lambda x: x["score"], reverse=True)

    print("=== Top 5 Hydrate Demo Candidates (class 8 real) ===")
    for i, c in enumerate(hydrate_candidates[:5], 1):
        print(f"  {i}. {c['instance_id']} (well {c['well_id']}, score={c['score']})")
        print(f"     {', '.join(c['reasons'])}")

    print(f"\n=== Top 3 Scaling Demo Candidates (class 7 real) ===")
    for i, c in enumerate(scaling_candidates[:3], 1):
        print(f"  {i}. {c['instance_id']} (well {c['well_id']}, score={c['score']})")
        print(f"     {', '.join(c['reasons'])}")

    # Write candidates JSON
    DATA_DEMO.mkdir(parents=True, exist_ok=True)
    output = {
        "hydrate": hydrate_candidates[:5],
        "scaling": scaling_candidates[:3],
    }
    out_path = DATA_DEMO / "candidates.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
