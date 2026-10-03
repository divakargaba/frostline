"""3W dataset loader and preprocessing.

Owner: Data+ML

Reads Petrobras 3W parquet files, downsamples to 1-minute resolution,
converts pressure from Pa to bar, tags each row with its source type
(real / simulated / drawn) and well_id, and derives the phase label:
  - normal
  - forming (original class + 100)
  - established

TODO:
  - Implement parquet discovery and loading
  - Implement downsampling logic (1-min mean)
  - Implement Pa -> bar conversion
  - Implement source tagging from file path conventions
  - Implement phase derivation from class labels
"""

import pandas as pd


def discover_parquet_files(raw_dir: str = "data/raw/3W") -> list[str]:
    """Find all parquet files in the 3W dataset directory.

    Args:
        raw_dir: Path to the raw 3W dataset root.

    Returns:
        List of absolute paths to parquet files.
    """
    raise NotImplementedError


def load_well(parquet_path: str) -> pd.DataFrame:
    """Load a single well's parquet file and apply preprocessing.

    Preprocessing steps:
      1. Read parquet
      2. Downsample to 1-minute resolution
      3. Convert pressure columns from Pa to bar
      4. Tag source type and well_id
      5. Derive phase column

    Args:
        parquet_path: Path to a single well parquet file.

    Returns:
        Preprocessed DataFrame.
    """
    raise NotImplementedError


def load_all(raw_dir: str = "data/raw/3W") -> pd.DataFrame:
    """Load and concatenate all wells from the 3W dataset.

    Args:
        raw_dir: Path to the raw 3W dataset root.

    Returns:
        Combined DataFrame for all wells.
    """
    raise NotImplementedError
