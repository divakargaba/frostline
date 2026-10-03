"""3W dataset loader and preprocessing.

Owner: Data+ML

Reads Petrobras 3W parquet files (data/raw/3W/dataset/<class>/<file>.parquet),
downsamples each to 1-minute resolution, converts pressures from Pa to bar,
tags each row with source / well_id / instance_id / event_class / phase and
per-minute sensor availability flags, and writes one parquet per instance to
data/processed/ plus data/processed/index.csv.

Phase from the 3W class label:
  - normal       class == 0
  - forming      class >= 100 (transient, e.g. 108)
  - established  class 1-9   (steady state, e.g. 8)

CLI:
  python -m src.load --classes 0 4 6 7 8 9 --workers 4 --skip-existing
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW_DIR = "data/raw/3W/dataset"
OUT_DIR = "data/processed"
LOG_PATH = "logs/load.log"

SENSORS = [
    "P-PDG", "T-PDG", "P-TPT", "T-TPT", "P-MON-CKP",
    "P-JUS-CKP", "T-JUS-CKP", "ABER-CKP", "QGL",
]
PRESSURE_SENSORS = [s for s in SENSORS if s.startswith("P-")]
LABELS = ["class", "state"]
PA_PER_BAR = 1e5

INDEX_COLUMNS = [
    "instance_id", "well_id", "source", "event_class", "n_minutes",
    "minutes_normal", "minutes_forming", "minutes_established",
] + [f"has_{s}" for s in SENSORS] + ["frozen_sensors"]

log = logging.getLogger("frostline.load")


# --------------------------------------------------------------------------
# Tagging helpers
# --------------------------------------------------------------------------

def parse_source(filename: str) -> tuple[str, str | None]:
    """Return (source, well_id) from a 3W filename.

    WELL-00019_20170101120000.parquet -> ("real", "WELL-00019")
    SIMULATED_00001.parquet           -> ("simulated", None)
    DRAWN_00001.parquet               -> ("drawn", None)
    """
    stem = Path(filename).stem
    if stem.startswith("WELL-"):
        return "real", stem.split("_")[0]
    if stem.startswith("SIMULATED"):
        return "simulated", None
    if stem.startswith("DRAWN"):
        return "drawn", None
    raise ValueError(f"Unrecognised 3W filename: {filename}")


def make_instance_id(path: Path) -> str:
    """Unique id for a raw file.

    Real files keep their stem (WELL-00019_20170101120000). SIMULATED_/DRAWN_
    stems repeat across class folders, so the class is appended:
    dataset/8/SIMULATED_00001.parquet -> SIMULATED_00001_c8.
    """
    path = Path(path)
    if path.stem.startswith("WELL-"):
        return path.stem
    return f"{path.stem}_c{int(path.parent.name)}"


def phase_from_class(cls: pd.Series) -> pd.Series:
    """Map 3W class labels to normal / forming / established (NaN -> None)."""
    c = cls.astype("float64").to_numpy()
    phase = np.select(
        [c == 0, (c >= 1) & (c < 100), c >= 100],
        ["normal", "established", "forming"],
        default=None,
    )
    return pd.Series(phase, index=cls.index, dtype="object")


def _minute_mode(labels: pd.Series, minutes: pd.DatetimeIndex) -> pd.Series:
    """Most frequent non-NaN label per minute (ties -> smallest label)."""
    s = labels.dropna()
    if s.empty:
        return pd.Series(np.nan, index=minutes, dtype="float64")
    counts = (
        pd.DataFrame({"minute": s.index.floor("1min"), "label": s.to_numpy()})
        .groupby(["minute", "label"]).size().rename("n").reset_index()
        .sort_values(["minute", "n", "label"], ascending=[True, False, True])
        .drop_duplicates("minute")
        .set_index("minute")["label"]
    )
    return counts.reindex(minutes).astype("float64")


# --------------------------------------------------------------------------
# Single-file processing
# --------------------------------------------------------------------------

def load_well(parquet_path: str | os.PathLike) -> pd.DataFrame:
    """Load one raw 3W parquet file and return the 1-minute processed frame.

    Steps: read core columns (missing -> NaN), Pa -> bar, 1-min downsample
    (mean for sensors, mode for class/state), drop minutes with NaN class,
    add instance_id / well_id / source / event_class / phase / has_<sensor>.
    """
    path = Path(parquet_path)
    source, well_id = parse_source(path.name)
    event_class = int(path.parent.name)

    available = set(pq.read_schema(path).names)
    wanted = [c for c in SENSORS + LABELS if c in available]
    raw = pd.read_parquet(path, columns=wanted)
    if not isinstance(raw.index, pd.DatetimeIndex):
        raw = raw.set_index("timestamp") if "timestamp" in raw.columns else raw
        raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    for col in SENSORS + LABELS:
        if col not in raw.columns:
            raw[col] = np.nan

    sensors = raw[SENSORS].astype("float64")
    # A sensor stuck at one value for the whole file (often 0.0) is dead, not
    # data. These are far more common in normal files than hydrate files, so
    # keeping them would let a model learn "frozen sensor = normal".
    frozen = [c for c in SENSORS if sensors[c].nunique(dropna=True) == 1]
    sensors[frozen] = np.nan
    sensors[PRESSURE_SENSORS] = sensors[PRESSURE_SENSORS] / PA_PER_BAR
    out = sensors.resample("1min").mean()
    for label in LABELS:
        out[label] = _minute_mode(raw[label].astype("float64"), out.index)
    del raw, sensors

    out = out[out["class"].notna()].copy()
    out["class"] = out["class"].astype("Int64")
    out["state"] = out["state"].round().astype("Int64")
    out.index.name = "timestamp"
    out.attrs["frozen_sensors"] = frozen

    out["instance_id"] = make_instance_id(path)
    out["well_id"] = well_id
    out["source"] = source
    out["event_class"] = event_class
    out["phase"] = phase_from_class(out["class"])
    for s in SENSORS:
        out[f"has_{s}"] = out[s].notna()
    return out


def summarize_instance(df: pd.DataFrame, path: Path) -> dict:
    """One index.csv row for a processed instance."""
    source, well_id = parse_source(path.name)
    phase_counts = df["phase"].value_counts()
    row = {
        "instance_id": make_instance_id(path),
        "well_id": well_id,
        "source": source,
        "event_class": int(path.parent.name),
        "n_minutes": len(df),
        "minutes_normal": int(phase_counts.get("normal", 0)),
        "minutes_forming": int(phase_counts.get("forming", 0)),
        "minutes_established": int(phase_counts.get("established", 0)),
    }
    for s in SENSORS:
        row[f"has_{s}"] = bool(df[s].notna().any())
    row["frozen_sensors"] = ";".join(df.attrs.get("frozen_sensors", []))
    return row


def process_file(raw_path: str, out_dir: str, skip_existing: bool) -> dict:
    """Worker: process one file. Never raises; returns a status dict."""
    path = Path(raw_path)
    out_path = Path(out_dir) / f"{make_instance_id(path)}.parquet"
    t0 = time.time()
    try:
        if skip_existing and out_path.exists():
            df = pd.read_parquet(out_path, columns=SENSORS + ["phase"])
            return {"status": "skipped", "path": raw_path,
                    "row": summarize_instance(df, path), "secs": time.time() - t0}
        df = load_well(path)
        tmp = out_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        os.replace(tmp, out_path)  # atomic: a crash never leaves a half file
        row = summarize_instance(df, path)
        del df
        return {"status": "ok", "path": raw_path, "row": row, "secs": time.time() - t0}
    except Exception:
        return {"status": "error", "path": raw_path,
                "error": traceback.format_exc(), "secs": time.time() - t0}


# --------------------------------------------------------------------------
# Batch run
# --------------------------------------------------------------------------

def discover_parquet_files(raw_dir: str = RAW_DIR,
                           classes: list[int] | None = None) -> list[Path]:
    """All raw parquet files under raw_dir/<class>/, optionally filtered."""
    root = Path(raw_dir)
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.isdigit())
    if classes is not None:
        dirs = [d for d in dirs if int(d.name) in classes]
    return [f for d in dirs for f in sorted(d.glob("*.parquet"))]


def check_unique_ids(files: list[Path]) -> None:
    """Each file must map to its own output parquet."""
    seen: dict[str, Path] = {}
    dupes = []
    for f in files:
        iid = make_instance_id(f)
        if iid in seen:
            dupes.append((seen[iid], f))
        seen[iid] = f
    if dupes:
        examples = "; ".join(f"{a} vs {b}" for a, b in dupes[:5])
        raise SystemExit(f"{len(dupes)} duplicate instance_ids across folders, e.g. {examples}")


def write_index(rows: list[dict], out_dir: str) -> pd.DataFrame:
    """Merge this run's rows into out_dir/index.csv (other classes are kept)."""
    index_path = Path(out_dir) / "index.csv"
    new = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    if index_path.exists():
        old = pd.read_csv(index_path)
        old = old[~old["instance_id"].isin(new["instance_id"])]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["event_class", "source", "instance_id"]).reset_index(drop=True)
    new.to_csv(index_path, index=False)
    return new


def print_summary(index: pd.DataFrame) -> None:
    """Instances per event_class x source, sensor availability %, minutes per phase."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n=== Instances per event_class x source ===")
    print(pd.crosstab(index["event_class"], index["source"], margins=True, margins_name="total"))

    print("\n=== Sensor availability (% of instances), by event_class x source ===")
    has_cols = [f"has_{s}" for s in SENSORS]
    avail = (index.groupby(["event_class", "source"])[has_cols].mean() * 100).round(0)
    avail.columns = SENSORS
    print(avail.astype(int))

    print("\n=== Total minutes per phase, by event_class x source ===")
    mins = index.groupby(["event_class", "source"])[
        ["minutes_normal", "minutes_forming", "minutes_established"]].sum()
    mins.loc[("all", ""), :] = mins.sum()
    print(mins.astype(int))


def _setup_logging(log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.handlers[:] = [fh, sh]
    log.setLevel(logging.INFO)


def load_all(raw_dir: str = RAW_DIR, out_dir: str = OUT_DIR,
             classes: list[int] | None = None, workers: int = 4,
             skip_existing: bool = False) -> pd.DataFrame:
    """Process every raw file to out_dir and return the updated index."""
    files = discover_parquet_files(raw_dir, classes)
    check_unique_ids(files)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log.info("Processing %d files (classes=%s, workers=%d, skip_existing=%s)",
             len(files), classes, workers, skip_existing)

    rows, n_err, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_file, str(f), out_dir, skip_existing) for f in files]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            name = Path(res["path"]).name
            if res["status"] == "error":
                n_err += 1
                log.error("[%d/%d] FAILED %s\n%s", i, len(files), res["path"], res["error"])
                continue
            rows.append(res["row"])
            log.info("[%d/%d] %s %s (%d min, %.1fs)", i, len(files), res["status"],
                     name, res["row"]["n_minutes"], res["secs"])

    log.info("Done: %d ok/skipped, %d errors in %.0fs", len(rows), n_err, time.time() - t0)
    return write_index(rows, out_dir)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="3W raw parquet -> 1-min processed parquet")
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                    help="event class folders to process (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse already-processed instances (resume after a crash)")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--log", default=LOG_PATH)
    args = ap.parse_args(argv)

    _setup_logging(args.log)
    index = load_all(args.raw_dir, args.out_dir, args.classes, args.workers, args.skip_existing)
    print_summary(index)


if __name__ == "__main__":
    main()
