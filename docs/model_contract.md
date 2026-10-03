# Model Contract

Expected artifact from Data+ML teammate for the LightGBM classifier.

## Directory Structure

```
models/frostline_v<N>/
  model.txt                   # LightGBM model file (Booster.save_model)
  metadata.json               # See below
```

## metadata.json

```json
{
  "version": 1,
  "created_at": "2026-10-03T12:00:00",
  "feature_order": [
    "P-PDG_mean_10min", "P-PDG_std_10min", "P-PDG_slope_10min",
    "..."
  ],
  "preprocessing_version": "v1",
  "training_well_ids": ["WELL-00025", "WELL-00026", "..."],
  "dataset_fingerprint": "sha256 of training data index",
  "classes": ["normal", "hydrate", "lookalike"],
  "class_map": {"0": "normal", "1": "hydrate", "2": "lookalike"},
  "alarm_threshold": 0.5,
  "alarm_persistence_min": 3,
  "notes": "Optional notes about training"
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `version` | yes | Integer version number |
| `created_at` | yes | ISO timestamp |
| `feature_order` | yes | Exact column names the model expects, in order |
| `preprocessing_version` | no | Which feature pipeline version was used |
| `training_well_ids` | no | Wells used for training |
| `dataset_fingerprint` | no | Hash of training data for reproducibility |
| `classes` | yes | Class names in order. 2 classes (binary) or 3 classes supported |
| `class_map` | yes | Maps model output index to class name |
| `alarm_threshold` | no | Watcher threshold for p_hydrate (default: 0.5) |
| `alarm_persistence_min` | no | Consecutive minutes above threshold to trigger (default: 3) |
| `notes` | no | Free text |

## predict() Contract

The ModelAdapter in `src/tools.py` calls `model.predict(feature_array)` and maps outputs:

- **3-class model** (`classes: [normal, hydrate, lookalike]`):
  Returns `{p_hydrate, p_lookalike, p_normal}` from `predict_proba()`.

- **2-class (binary) model** (`classes: [normal, hydrate]`):
  Returns `{p_hydrate, p_normal}`, sets `p_lookalike = 0`.

Output is labeled `source: "model_score"` (not "probability") since LightGBM scores are not calibrated.

## Loading

`ModelAdapter` auto-loads the newest `models/frostline_v*/` at startup by sorting version numbers. If the model is missing, invalid, or features don't match, `classify_event` falls back to the heuristic with `source: "heuristic_fallback"`, and `/health` reports `has_model: false` with the reason.

## Watcher Integration

If `metadata.json` includes `alarm_threshold` and `alarm_persistence_min`, the watcher uses them instead of defaults from `DEFAULT_THRESHOLDS`.
