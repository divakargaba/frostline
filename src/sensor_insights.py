"""Per-sensor change readings and combined pressure/temperature inferences.

Uses only the observed window (never labels or future rows). These are review
observations that explain *why* a change matters; they do not change the model
score or the alarm policy, and they are not diagnoses.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

WINDOW_MINUTES = 10

SENSOR_INFO = [
    {"code": "P-PDG", "name": "Downhole pressure", "location": "Permanent downhole gauge, bottom of the production tubing", "unit": "bar",
     "why": "Deepest pressure. Compared with P-TPT it shows whether flow is being held back inside the tubing; a blockage above the gauge tends to raise it."},
    {"code": "T-PDG", "name": "Downhole temperature", "location": "Permanent downhole gauge", "unit": "°C",
     "why": "Reservoir-side temperature; the reference for how much the fluid cools on the way up."},
    {"code": "P-TPT", "name": "Tree pressure", "location": "Subsea Christmas tree (temperature/pressure transducer)", "unit": "bar",
     "why": "Main subsea pressure. Its difference to P-PDG (tubing) and to P-MON-CKP (production line) locates a restriction."},
    {"code": "T-TPT", "name": "Tree temperature", "location": "Subsea Christmas tree", "unit": "°C",
     "why": "Coldest point most exposed to seawater. Falling temperature at rising pressure moves the fluid toward hydrate-forming conditions."},
    {"code": "P-MON-CKP", "name": "Pre-choke pressure", "location": "Topside, upstream of the production choke", "unit": "bar",
     "why": "End of the production line. A growing P-TPT minus P-MON-CKP gap points to a restriction in the line (hydrate or scale candidates)."},
    {"code": "T-MON-CKP", "name": "Pre-choke temperature", "location": "Topside, upstream of the production choke", "unit": "°C",
     "why": "Backs up T-TPT when it is missing; cooling here means less warm fluid is arriving."},
    {"code": "P-JUS-CKP", "name": "Post-choke pressure", "location": "Topside, downstream of the production choke", "unit": "bar",
     "why": "With P-MON-CKP gives the drop across the choke; a changing drop with a fixed choke opening suggests a choke restriction."},
    {"code": "T-JUS-CKP", "name": "Post-choke temperature", "location": "Topside, downstream of the production choke", "unit": "°C",
     "why": "Expansion cooling across the choke; often missing in the real data."},
    {"code": "ABER-CKP", "name": "Choke opening", "location": "Production choke", "unit": "%",
     "why": "Explains operational pressure changes. Pressure moving while the choke is unchanged is not explained by an adjustment."},
    {"code": "QGL", "name": "Gas-lift flow", "location": "Gas-lift injection line", "unit": "m³/s",
     "why": "Gas-lift injection, not oil production. Sudden changes shift pressures and flow and can mimic a process event."},
]
INFO = {item["code"]: item for item in SENSOR_INFO}
PAIRS = [("tubing", "P-PDG", "P-TPT", "downhole-to-tree", "a restriction in the production tubing"),
         ("line", "P-TPT", "P-MON-CKP", "tree-to-choke", "a restriction in the production line (hydrate or scale candidates)"),
         ("choke", "P-MON-CKP", "P-JUS-CKP", "across-choke", "a restriction at the choke")]


def _clean(frame):
    frame = frame.reindex(columns=list(INFO)).astype(float).iloc[-(WINDOW_MINUTES + 1):]
    values = frame.to_numpy(copy=True)
    values[~np.isfinite(values) | (np.abs(values) >= 1e25)] = np.nan
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def _pressure_step(reference):
    return max(1.0, abs(reference or 0) * .01)


def _round(value, digits=2):
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def _channel(code, series):
    info = INFO[code]
    valid = series.dropna()
    current = series.iloc[-1] if len(series) else float("nan")
    out = {"code": code, "name": info["name"], "unit": info["unit"], "value": _round(current), "change": None, "trend": "unavailable", "notable": False, "note": ""}
    if pd.isna(current):
        out["note"] = "No valid reading; this channel cannot support an inference."
        return out
    if code.startswith(("P-", "T-")) and len(valid) >= 3 and valid.nunique() == 1 and valid.iloc[-1] == 0:
        out.update(trend="inactive", note=f"Constant 0 {info['unit']}: the gauge looks inactive, so it is not process evidence.")
        return out
    if len(valid) < 3:
        out["note"] = "Too few readings in the window to judge a trend."
        return out
    change = float(valid.iloc[-1] - valid.iloc[0])
    if code.startswith("P-"):
        step = _pressure_step(valid.iloc[0])
    elif code.startswith("T-"):
        step = 1.0
    elif code == "ABER-CKP":
        step = .5
    else:
        step = max(.01, abs(valid.iloc[0]) * .1)
    notable = abs(change) >= step
    trend = ("rising" if change > 0 else "falling") if notable else "steady"
    out.update(change=_round(change), trend=trend, notable=bool(notable))
    if code in ("P-PDG", "P-TPT", "P-MON-CKP", "P-JUS-CKP") and notable:
        out["note"] = f"{info['name']} {trend} {abs(change):.2f} bar in {WINDOW_MINUTES} min."
    elif code.startswith("T-") and notable:
        out["note"] = f"{info['name']} {trend} {abs(change):.2f} °C in {WINDOW_MINUTES} min" + ("; colder fluid is closer to hydrate-forming conditions." if change < 0 else ".")
    elif code == "ABER-CKP":
        out["note"] = f"Choke moved {change:+.2f} percentage points; pressure changes may be operational." if notable else "Choke opening unchanged."
    elif code == "QGL" and notable:
        out["note"] = f"Gas-lift rate {trend} {abs(change):.3f} m³/s; check whether it was planned."
    else:
        out["note"] = "Steady within the window."
    return out


def sensor_insights(observed_frame):
    """Readings and inferences for the last WINDOW_MINUTES of an observed prefix."""
    if observed_frame is None or len(observed_frame) == 0:
        return {"as_of": None, "window_minutes": WINDOW_MINUTES, "channels": [], "inferences": []}
    frame = _clean(observed_frame)
    channels = {code: _channel(code, frame[code]) for code in INFO}
    usable = lambda code: channels[code]["trend"] not in ("unavailable", "inactive")
    inferences = []

    def add(ident, level, sensors, text, why):
        inferences.append({"id": ident, "level": level, "sensors": sensors, "text": text, "why": why})

    for ident, a, b, label, meaning in PAIRS:
        if not (usable(a) and usable(b)):
            continue
        diff = (frame[a] - frame[b]).dropna()
        if len(diff) < 3:
            continue
        change = float(diff.iloc[-1] - diff.iloc[0])
        if abs(change) >= _pressure_step(diff.iloc[0]):
            grew = abs(diff.iloc[-1]) > abs(diff.iloc[0])
            add(f"{ident}_differential", "watch" if grew else "info", [a, b],
                f"The {label} pressure difference ({a} − {b}) {'grew' if grew else 'shrank'} by {abs(change):.2f} bar to {diff.iloc[-1]:.2f} bar.",
                f"A growing difference means more pressure is lost along that section, consistent with {meaning}." if grew
                else "A shrinking difference means less pressure is lost along that section; flow resistance there is easing or flow has dropped.")

    for t_code, p_code, where in [("T-TPT", "P-TPT", "the tree"), ("T-MON-CKP", "P-MON-CKP", "the choke inlet")]:
        t, p = channels[t_code], channels[p_code]
        if not (usable(t_code) and usable(p_code)) or not t["notable"]:
            continue
        if t["trend"] == "falling" and p["trend"] == "rising":
            add(f"hydrate_conditions_{p_code}", "watch", [t_code, p_code],
                f"At {where}, temperature fell {abs(t['change']):.2f} °C while pressure rose {abs(p['change']):.2f} bar.",
                "Hydrates are stable at high pressure and low temperature, so this combination moves the fluid toward hydrate-forming conditions.")
        elif t["trend"] == "falling" and p["trend"] == "falling":
            add(f"flow_loss_{p_code}", "watch", [t_code, p_code],
                f"At {where}, temperature fell {abs(t['change']):.2f} °C and pressure fell {abs(p['change']):.2f} bar together.",
                "Less warm fluid is arriving: flow has dropped, from a restriction upstream or an operating change. Cooling of a stalled line raises hydrate risk.")
        elif t["trend"] == "falling":
            add(f"cooling_{t_code}", "info", [t_code],
                f"Temperature at {where} fell {abs(t['change']):.2f} °C with steady pressure.",
                "Cooling alone narrows the margin to hydrate-forming conditions; watch whether pressure follows.")

    moved = [code for code in ("P-TPT", "P-MON-CKP", "P-JUS-CKP") if usable(code) and channels[code]["notable"]]
    if moved and usable("ABER-CKP"):
        if channels["ABER-CKP"]["notable"]:
            add("choke_moved", "info", ["ABER-CKP", *moved], f"The choke moved {channels['ABER-CKP']['change']:+.2f} pp while {', '.join(moved)} changed.",
                "A choke adjustment is a likely operational explanation for the pressure change.")
        else:
            add("pressure_without_choke", "watch", ["ABER-CKP", *moved], f"{', '.join(moved)} changed while the choke opening stayed fixed.",
                "The change is not explained by a choke adjustment, so a process cause (restriction, flow change) needs checking.")
    if usable("QGL") and channels["QGL"]["notable"]:
        add("gas_lift_change", "info", ["QGL"], f"Gas-lift flow changed by {channels['QGL']['change']:+.3f} m³/s.",
            "Gas-lift changes shift tubing pressure and flow; confirm whether the change was planned before attributing pressure moves to a process event.")
    inactive = [code for code, ch in channels.items() if ch["trend"] == "inactive"]
    if inactive:
        add("inactive_gauges", "info", inactive, f"{', '.join(inactive)} read a constant 0.",
            "Inactive gauges are excluded from these inferences; comparisons that need them are unavailable.")
    order = {"watch": 0, "info": 1}
    inferences.sort(key=lambda item: order[item["level"]])
    return {"as_of": observed_frame.index[-1].isoformat(), "window_minutes": WINDOW_MINUTES,
            "channels": list(channels.values()), "inferences": inferences,
            "note": "Observations from measured changes; not a diagnosis or a hydrate equilibrium calculation."}
