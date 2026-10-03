"""Agent tool definitions with JSON schemas and fallbacks.

Owner: Div

Each tool is a plain function taking an AgentContext + tool-specific args,
returning compact JSON. Tools never expose the phase label (that's what
the agent is trying to predict). Every result includes "available" and "source".

HARD RULE: tools may only see data up to and including ctx.minute_index.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("frostline.tools")

THRESHOLDS_PATH = Path("data/state/thresholds.json")
MODELS_DIR = Path("models")
TRIGGER_LOG_PATH = Path("data/state/trigger_log.jsonl")

DEFAULT_THRESHOLDS = {
    "watch_threshold": 0.5,
    "margin_warn_band_C": 3.0,
    "pressure_sigma_k": 8.0,              # Trigger when 30/60-min change > k * rolling_std
    "pressure_sigma_sustain_min": 3,       # Must exceed k-sigma for N consecutive minutes
    "pressure_abs_floor_bar": 0.5,         # Minimum absolute change (prevents noise on flat sensors)
    "lookalike_threshold": 0.5,            # p_lookalike trigger threshold (when model exists)
    "cooldown_min": 60,
    "recheck_default_min": 15,
}


def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        return json.loads(THRESHOLDS_PATH.read_text())
    return dict(DEFAULT_THRESHOLDS)


def save_thresholds(thresholds: dict) -> None:
    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))


@dataclass
class AgentContext:
    """Everything a tool needs — the instance data up to the current minute."""
    instance_id: str
    well_id: str
    minute_index: int
    df: pd.DataFrame  # Full enriched frame (tools MUST slice to [:minute_index+1])
    decisions: list[dict] = field(default_factory=list)  # Past decisions this run
    thresholds: dict = field(default_factory=load_thresholds)

    @property
    def window(self) -> pd.DataFrame:
        """Data up to and including current minute — enforces causality."""
        return self.df.iloc[:self.minute_index + 1]

    @property
    def current(self) -> pd.Series:
        return self.df.iloc[self.minute_index]


# ---------------------------------------------------------------------------
# Model adapter — auto-loads newest models/frostline_v*/
# ---------------------------------------------------------------------------

class ModelAdapter:
    """Loads a LightGBM model from models/frostline_v<N>/ or models/lgbm_hydrate.pkl."""

    def __init__(self):
        self.model = None
        self.metadata: dict = {}
        self.feature_order: list[str] = []
        self.classes: list[str] = []
        self.class_map: dict[str, str] = {}
        self.ready = False
        self.error: str = ""
        self.source: str = ""
        self._load()

    def _load(self):
        if not MODELS_DIR.exists():
            self.error = "models/ directory not found"
            return

        # Try frostline_v*/ first (contract format)
        versions = sorted(MODELS_DIR.glob("frostline_v*/"), key=lambda p: p.name)
        if versions:
            self._load_contract(versions[-1])
            if self.ready:
                return

        # Fall back to Zaine's joblib pkl via src.model.load_model
        pkl_path = MODELS_DIR / "lgbm_hydrate.pkl"
        if pkl_path.exists():
            self._load_pkl(pkl_path)
            return

        if not self.error:
            self.error = "no model found (no frostline_v*/ or lgbm_hydrate.pkl)"

    def _load_contract(self, model_dir: Path):
        """Load from models/frostline_v<N>/ (model.txt + metadata.json)."""
        meta_path = model_dir / "metadata.json"
        model_path = model_dir / "model.txt"

        if not meta_path.exists():
            self.error = f"metadata.json missing in {model_dir.name}"
            return
        if not model_path.exists():
            self.error = f"model.txt missing in {model_dir.name}"
            return

        try:
            self.metadata = json.loads(meta_path.read_text())
        except Exception as e:
            self.error = f"metadata.json invalid: {e}"
            return

        self.feature_order = self.metadata.get("feature_order", [])
        self.classes = self.metadata.get("classes", [])
        self.class_map = self.metadata.get("class_map", {})

        if not self.feature_order:
            self.error = "metadata.json missing feature_order"
            return
        if not self.classes or len(self.classes) < 2:
            self.error = "metadata.json missing or invalid classes (need 2+)"
            return

        try:
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(model_path))
            self.ready = True
            self.source = f"contract:{model_dir.name}"
            log.info("Loaded model from %s (%d features, %d classes)",
                     model_dir.name, len(self.feature_order), len(self.classes))
        except ImportError:
            self.error = "lightgbm not installed"
        except Exception as e:
            self.error = f"model load failed: {e}"

    def _load_pkl(self, pkl_path: Path):
        """Load from Zaine's joblib bundle (models/lgbm_hydrate.pkl)."""
        try:
            from src.model import load_model, FINE_CLASSES, GROUPS
            bundle = load_model(str(pkl_path))
            self.model = bundle
            self.feature_order = bundle.get("features", [])
            self.classes = bundle.get("classes", FINE_CLASSES)
            self.metadata = {
                "source": "lgbm_hydrate.pkl",
                "n_features": len(self.feature_order),
                "classes": self.classes,
            }
            self.ready = True
            self.source = "pkl:lgbm_hydrate.pkl"
            log.info("Loaded model from lgbm_hydrate.pkl (%d features, %d classes)",
                     len(self.feature_order), len(self.classes))
        except Exception as e:
            self.error = f"pkl load failed: {e}"

    def predict(self, window_df: pd.DataFrame) -> dict | None:
        """Predict on the last row of window_df. Returns {p_hydrate, p_lookalike, p_normal}."""
        if not self.ready or self.model is None:
            return None

        # pkl bundle: use src.model.predict which handles features + smoothing
        if self.source.startswith("pkl:"):
            try:
                from src.model import predict as _predict
                out = _predict(window_df, model=self.model)
                last = out.iloc[-1]
                p_h = last.get("p_hydrate")
                if p_h is None or (isinstance(p_h, float) and np.isnan(p_h)):
                    return None
                return {
                    "p_hydrate": float(last["p_hydrate"]),
                    "p_lookalike": float(last.get("p_lookalike", 0)),
                    "p_normal": float(last.get("p_normal", 0)),
                }
            except Exception as e:
                log.error("pkl predict failed: %s", e)
                return None

        # Contract format: build feature array from the last row
        last = window_df.iloc[-1:]
        missing = [f for f in self.feature_order if f not in last.columns]
        if missing:
            log.warning("Model missing features: %s", missing[:5])
            return None

        X = last[self.feature_order].values
        try:
            proba = self.model.predict(X)
        except Exception as e:
            log.error("Model predict failed: %s", e)
            return None

        # Map to standard output
        result: dict[str, float] = {"p_hydrate": 0.0, "p_lookalike": 0.0, "p_normal": 0.0}
        if len(self.classes) == 2:
            if proba.ndim == 1:
                score = float(proba[0])
            else:
                score = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
            result["p_hydrate"] = score
            result["p_normal"] = 1.0 - score
        elif len(self.classes) >= 3:
            if proba.ndim == 1:
                proba = proba.reshape(1, -1)
            for i, cls in enumerate(self.classes):
                mapped = self.class_map.get(str(i), cls).lower()
                if "hydrate" in mapped:
                    result["p_hydrate"] = float(proba[0][i])
                elif "lookalike" in mapped or "look" in mapped:
                    result["p_lookalike"] = float(proba[0][i])
                elif "normal" in mapped:
                    result["p_normal"] = float(proba[0][i])
        return result

    def watcher_overrides(self) -> dict:
        """Return threshold overrides from model metadata."""
        overrides = {}
        if "alarm_threshold" in self.metadata:
            overrides["watch_threshold"] = self.metadata["alarm_threshold"]
        if "alarm_persistence_min" in self.metadata:
            # Not directly in DEFAULT_THRESHOLDS but watcher can use it
            overrides["alarm_persistence_min"] = self.metadata["alarm_persistence_min"]
        return overrides


# Global model adapter instance
_model_adapter: ModelAdapter | None = None


def get_model_adapter() -> ModelAdapter:
    global _model_adapter
    if _model_adapter is None:
        _model_adapter = ModelAdapter()
    return _model_adapter


def _clean(v: Any) -> Any:
    """Make a value JSON-safe."""
    if v is None:
        return None
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _round(v: Any, n: int = 2) -> Any:
    if isinstance(v, float) and not (np.isnan(v) or np.isinf(v)):
        return round(v, n)
    return _clean(v)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_window(ctx: AgentContext, minutes: int = 60) -> dict:
    """Latest sensor values and slopes. Never exposes phase."""
    w = ctx.window
    tail = w.tail(min(minutes, len(w)))
    sensors = ["P-PDG", "T-PDG", "P-TPT", "T-TPT", "P-MON-CKP",
               "P-JUS-CKP", "T-JUS-CKP", "ABER-CKP", "QGL"]
    latest = {}
    missing = []
    for s in sensors:
        val = _clean(tail[s].iloc[-1]) if s in tail.columns else None
        if val is None:
            missing.append(s)
        latest[s] = _round(val)

    slopes = {}
    for win in [10, 30, 60]:
        t = tail.tail(min(win, len(tail)))
        if len(t) < 3:
            continue
        for s in sensors:
            if s not in t.columns or t[s].isna().all():
                continue
            vals = t[s].dropna()
            if len(vals) < 2:
                continue
            x = np.arange(len(vals), dtype=float)
            y = vals.values.astype(float)
            if np.std(x) < 1e-9:
                continue
            slope = float(np.polyfit(x, y, 1)[0])
            slopes[f"{s}_slope_{win}min"] = _round(slope, 4)

    return {
        "available": True,
        "source": "replay_data",
        "rows": len(tail),
        "latest": latest,
        "missing_sensors": missing,
        "slopes": slopes,
    }


def tool_hydrate_margin(ctx: AgentContext) -> dict:
    """Subcooling margin from physics module."""
    row = ctx.current
    p = _clean(row.get("P-TPT"))
    t = _clean(row.get("T-TPT"))

    if p is None or t is None:
        return {
            "available": False,
            "source": "physics",
            "reason": "P-TPT or T-TPT sensor missing",
        }

    try:
        from src.physics import hydrate_margin as _hm
        margin = _hm(float(p), float(t))
    except (ImportError, NotImplementedError):
        return {
            "available": False,
            "source": "physics",
            "reason": "src.physics.hydrate_margin not implemented yet",
        }

    # Margin slope from recent window
    w = ctx.window
    margin_slope = None
    if "margin_C" in w.columns:
        margins = w["margin_C"].dropna()
        if len(margins) >= 5:
            x = np.arange(len(margins[-10:]), dtype=float)
            y = margins.values[-10:].astype(float)
            margin_slope = _round(float(np.polyfit(x, y, 1)[0]), 4)

    status = "safe" if margin > 3 else ("warning" if margin > 0 else "danger")
    return {
        "available": True,
        "source": "physics",
        "margin_C": _round(margin),
        "margin_slope_C_per_min": margin_slope,
        "status": status,
        "out_of_range": False,
    }


def tool_classify_event(ctx: AgentContext) -> dict:
    """ML classification or heuristic fallback."""
    # Fast path: replay already scored the full history causally; use that minute's score.
    if {"p_hydrate", "p_lookalike", "p_normal"} <= set(ctx.window.columns):
        last = ctx.current
        if _clean(last.get("p_hydrate")) is not None:
            return {
                "available": True,
                "source": "ml_model",
                "p_hydrate": _round(_clean(last["p_hydrate"])),
                "p_lookalike": _round(_clean(last["p_lookalike"])),
                "p_normal": _round(_clean(last["p_normal"])),
            }

    # Try ModelAdapter (models/frostline_v*/)
    adapter = get_model_adapter()
    if adapter.ready:
        w = ctx.window.tail(60).copy()
        result = adapter.predict(w)
        if result is not None:
            return {
                "available": True,
                "source": "model_score",
                "p_hydrate": _round(result.get("p_hydrate", 0)),
                "p_lookalike": _round(result.get("p_lookalike", 0)),
                "p_normal": _round(result.get("p_normal", 0)),
            }

    # Direct src.model.predict call
    try:
        from src.model import predict as _predict
        w = ctx.window.tail(60).copy()
        result = _predict(w)
        if result is not None and "p_hydrate" in result.columns:
            last = result.iloc[-1]
            return {
                "available": True,
                "source": "model_score",
                "p_hydrate": _round(_clean(last.get("p_hydrate", 0))),
                "p_lookalike": _round(_clean(last.get("p_lookalike", 0))),
                "p_normal": _round(_clean(last.get("p_normal", 0))),
            }
    except (ImportError, NotImplementedError, Exception) as e:
        log.debug("ML classify fallback: %s", e)

    # Heuristic fallback
    w = ctx.window
    tail30 = w.tail(min(30, len(w)))

    p_tpt = tail30["P-TPT"].dropna() if "P-TPT" in tail30.columns else pd.Series(dtype=float)
    t_tpt = tail30["T-TPT"].dropna() if "T-TPT" in tail30.columns else pd.Series(dtype=float)

    p_drop = 0.0
    t_drop = 0.0
    if len(p_tpt) >= 2:
        p_drop = float(p_tpt.iloc[0] - p_tpt.iloc[-1])  # positive = dropped
    if len(t_tpt) >= 2:
        t_drop = float(t_tpt.iloc[0] - t_tpt.iloc[-1])

    # Heuristic: pressure + temperature both dropping -> hydrate
    # Pressure dropping but temp stable -> look-alike (scaling/restriction)
    if p_drop > 3 and t_drop > 1:
        p_h, p_l, p_n = 0.65, 0.15, 0.20
    elif p_drop > 3:
        p_h, p_l, p_n = 0.25, 0.50, 0.25
    elif p_drop > 1:
        p_h, p_l, p_n = 0.15, 0.25, 0.60
    else:
        p_h, p_l, p_n = 0.05, 0.10, 0.85

    return {
        "available": True,
        "source": "heuristic_fallback",
        "p_hydrate": _round(p_h),
        "p_lookalike": _round(p_l),
        "p_normal": _round(p_n),
        "note": "ML model not available; using pressure/temperature trend heuristic",
    }


def tool_forecast_onset(ctx: AgentContext) -> dict:
    """Onset forecast from ML or physics extrapolation."""
    # Try ML forecast
    try:
        from src.forecast import forecast_onset as _fo
        w = ctx.window.tail(60)
        result = _fo(w)
        if result is not None:
            return {
                "available": True,
                "source": "ml_forecast",
                "p10": _round(_clean(result.get("q10"))),
                "p50": _round(_clean(result.get("q50"))),
                "p90": _round(_clean(result.get("q90"))),
            }
    except (ImportError, NotImplementedError, Exception) as e:
        log.debug("ML forecast fallback: %s", e)

    # Physics extrapolation: if margin and margin_slope available
    margin_result = tool_hydrate_margin(ctx)
    if margin_result.get("available") and margin_result.get("margin_slope_C_per_min") is not None:
        margin = margin_result["margin_C"]
        slope = margin_result["margin_slope_C_per_min"]
        if slope < -0.001 and margin is not None:
            # Time for margin to reach some negative threshold (e.g. -5 C)
            target = -5.0
            mins_to_target = (target - margin) / slope
            return {
                "available": True,
                "source": "physics_extrapolation",
                "p10": _round(max(10, mins_to_target * 0.5)),
                "p50": _round(max(15, mins_to_target)),
                "p90": _round(max(30, mins_to_target * 2)),
                "note": f"Extrapolated from margin={margin:.1f}C, slope={slope:.4f}C/min",
            }

    return {
        "available": False,
        "source": "forecast",
        "reason": "No ML forecast model and insufficient physics data for extrapolation",
    }


def tool_methanol_dose(ctx: AgentContext, target_shift_C: float = 5.0) -> dict:
    """Methanol dose via Hammerschmidt equation."""
    try:
        from src.physics import hammerschmidt_dose
        dose = hammerschmidt_dose(target_shift_C, inhibitor="methanol")
        in_range = dose < 25.0
        return {
            "available": True,
            "source": "physics",
            "dose_wt_pct": _round(dose),
            "target_shift_C": _round(target_shift_C),
            "in_range": in_range,
            "note": "Injection rate requires water rate (not in 3W data)" if in_range else "Dose exceeds typical range (>25 wt%)",
        }
    except (ImportError, NotImplementedError):
        return {
            "available": False,
            "source": "physics",
            "reason": "src.physics.hammerschmidt_dose not implemented yet",
        }


def tool_well_history(ctx: AgentContext) -> dict:
    """Past triggers/decisions for this well."""
    thresholds = ctx.thresholds
    decisions = ctx.decisions[-10:]  # Last 10

    # Also check trigger log
    trigger_history = []
    if TRIGGER_LOG_PATH.exists():
        for line in TRIGGER_LOG_PATH.read_text().splitlines()[-20:]:
            try:
                entry = json.loads(line)
                if entry.get("well_id") == ctx.well_id:
                    trigger_history.append(entry)
            except json.JSONDecodeError:
                continue

    return {
        "available": True,
        "source": "state",
        "well_id": ctx.well_id,
        "current_thresholds": thresholds,
        "past_decisions": decisions,
        "recent_triggers": trigger_history[-5:],
    }


def tool_check_sensor_quality(ctx: AgentContext) -> dict:
    """Check data quality for all sensors at current minute."""
    import json as _json

    row = ctx.current
    quality_str = row.get("_quality", "{}")
    quality = _json.loads(quality_str) if isinstance(quality_str, str) else {}

    sensors_status = {}
    stuck_sensors = []

    for s in ["P-PDG", "T-PDG", "P-TPT", "T-TPT", "P-MON-CKP",
              "P-JUS-CKP", "T-JUS-CKP", "ABER-CKP", "QGL"]:
        val = _clean(row.get(s))
        flag = quality.get(s)
        status = "ok" if flag is None else flag
        sensors_status[s] = {"value": _round(val), "status": status}
        if flag == "stuck":
            # Count how long it's been stuck
            w = ctx.window
            if s in w.columns:
                vals = w[s].dropna()
                if len(vals) >= 2 and vals.nunique() == 1:
                    stuck_sensors.append({"sensor": s, "duration_min": len(vals)})
                else:
                    # Count trailing identical values
                    count = 1
                    for i in range(len(vals) - 2, -1, -1):
                        if vals.iloc[i] == vals.iloc[-1]:
                            count += 1
                        else:
                            break
                    stuck_sensors.append({"sensor": s, "duration_min": count})

    return {
        "available": True,
        "source": "quality_check",
        "sensors": sensors_status,
        "stuck_sensors": stuck_sensors,
        "flagged": [s for s, info in sensors_status.items() if info["status"] != "ok"],
    }


def tool_search_playbook(ctx: AgentContext, query: str = "hydrate response") -> dict:
    """RAG search over playbook docs."""
    try:
        from src.rag import search
        results = search(query, top_k=3)
        return {
            "available": True,
            "source": "rag",
            "results": results,
        }
    except (ImportError, NotImplementedError):
        return {
            "available": False,
            "source": "rag",
            "reason": "RAG index not built yet",
        }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def _dispatch(name: str, ctx: AgentContext, args: dict) -> dict:
    """Call a tool by name with validated args."""
    if name not in TOOL_REGISTRY:
        return {"available": False, "error": f"Unknown tool: {name}"}
    fn, schema = TOOL_REGISTRY[name]
    # Validate required args against schema
    props = schema.get("function", {}).get("parameters", {}).get("properties", {})
    required = schema.get("function", {}).get("parameters", {}).get("required", [])
    for r in required:
        if r not in args:
            return {"available": False, "error": f"Missing required argument: {r}"}
    try:
        if name == "get_window":
            return fn(ctx, minutes=args.get("minutes", 60))
        elif name == "hydrate_margin":
            return fn(ctx)
        elif name == "classify_event":
            return fn(ctx)
        elif name == "forecast_onset":
            return fn(ctx)
        elif name == "methanol_dose":
            return fn(ctx, target_shift_C=args.get("target_shift_C", 5.0))
        elif name == "well_history":
            return fn(ctx)
        elif name == "search_playbook":
            return fn(ctx, query=args.get("query", "hydrate response"))
        elif name == "check_sensor_quality":
            return fn(ctx)
        else:
            return fn(ctx, **args)
    except Exception as e:
        log.error("Tool %s error: %s", name, e)
        return {"available": False, "error": str(e)}


# Schemas for OpenRouter tool calling
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_window",
            "description": "Get the latest N minutes of sensor data with slopes and missing sensor info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {"type": "integer", "description": "Lookback window in minutes", "default": 60},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hydrate_margin",
            "description": "Calculate subcooling margin (distance from hydrate equilibrium temperature). Negative = hydrate risk.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_event",
            "description": "Classify the current event as hydrate, look-alike (scaling/restriction/flow instability), or normal. Returns probabilities.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_onset",
            "description": "Forecast minutes until established hydrate phase (p10/p50/p90 prediction interval).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "methanol_dose",
            "description": "Calculate required methanol dose (wt%) for a given temperature shift using Hammerschmidt equation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_shift_C": {"type": "number", "description": "Target temperature depression in Celsius", "default": 5.0},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "well_history",
            "description": "Get past triggers, decisions, and current thresholds for this well.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_playbook",
            "description": "Search operations playbook for hydrate response procedures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_sensor_quality",
            "description": "Check data quality for all sensors: flags nonfinite, extreme, physically impossible, or stuck values.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

TOOL_REGISTRY: dict[str, tuple] = {
    "get_window": (get_window, TOOL_SCHEMAS[0]),
    "hydrate_margin": (tool_hydrate_margin, TOOL_SCHEMAS[1]),
    "classify_event": (tool_classify_event, TOOL_SCHEMAS[2]),
    "forecast_onset": (tool_forecast_onset, TOOL_SCHEMAS[3]),
    "methanol_dose": (tool_methanol_dose, TOOL_SCHEMAS[4]),
    "well_history": (tool_well_history, TOOL_SCHEMAS[5]),
    "search_playbook": (tool_search_playbook, TOOL_SCHEMAS[6]),
    "check_sensor_quality": (tool_check_sensor_quality, TOOL_SCHEMAS[7]),
}


def dispatch_tool(name: str, ctx: AgentContext, args: dict) -> dict:
    """Public entry point for calling a tool by name."""
    return _dispatch(name, ctx, args)
