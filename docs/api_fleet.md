# Fleet API Contract

Fleet replay system for monitoring multiple wells on a shared clock.

## Endpoints

### `POST /fleet/sessions`

Create a new fleet replay session. Loads wells from `data/demo/fleet.json`.

**Request:**
```json
{
  "speed": 60,
  "agent": true,
  "mode": "live",
  "record": false
}
```

- `speed`: ticks per second (1-6000)
- `agent`: enable watcher + agent investigations
- `mode`: `"live"` (real LLM), `"cached"` (replay recording), `"cache_only"` (agent uses cache/fallback only)
- `record`: save all events to `data/demo/fleet_<timestamp>.jsonl`

**Response:**
```json
{
  "session_id": "a1b2c3d4",
  "n_wells": 4,
  "max_minutes": 180,
  "mode": "live"
}
```

### `POST /fleet/sessions/{id}/pause`

Pause the replay clock.

**Response:** `{"session_id": "...", "paused": true}`

### `POST /fleet/sessions/{id}/resume`

Resume the replay clock.

**Response:** `{"session_id": "...", "paused": false}`

### `GET /fleet/sessions/{id}/stream`

SSE event stream. Multiple subscribers supported. Late joiners get a catch-up `fleet_status` event.

### `POST /fleet/sessions/{id}/incidents/{incident_id}/ack`

Acknowledge an incident. Monitoring continues; escalation still possible.

**Response:**
```json
{"incident_id": "WELL-00019_34", "acknowledged": true}
```

---

## SSE Event Types

### `fleet_tick`

Emitted for each well at each clock tick.

```json
{
  "session_id": "a1b2c3d4",
  "well_id": "WELL-00019",
  "display_id": "Delta-19",
  "replay_minute": 34,
  "original_t": "2012-06-02T20:59:00",
  "sensors": {
    "P_TPT_bar": 279.6,
    "T_TPT_C": 82.3,
    "P_PDG_bar": 281.2,
    "T_PDG_C": 83.1,
    "P_MON_CKP_bar": 278.4,
    "QGL": 9.95
  },
  "margin_C": 2.1,
  "p_hydrate": 0.72,
  "p_lookalike": 0.15,
  "p_normal": 0.13,
  "quality": {"P-PDG": "stuck"}
}
```

`quality` field only present when sensors have issues. Possible values: `"nonfinite"`, `"extreme"`, `"below_absolute_zero"`, `"choke_out_of_range"`, `"stuck"`.

### `fleet_phase_marker`

Emitted when a well transitions between phases.

```json
{
  "session_id": "a1b2c3d4",
  "well_id": "WELL-00019",
  "display_id": "Delta-19",
  "t": "2012-06-02T20:59:00",
  "phase": "forming"
}
```

### `fleet_watch_trigger`

Emitted when the watcher detects an anomaly on a well.

```json
{
  "session_id": "a1b2c3d4",
  "well_id": "WELL-00019",
  "display_id": "Delta-19",
  "t": "2012-06-02T20:59:00",
  "reason": "p_hydrate >= 0.5 for 3+ minutes",
  "score": 0.72,
  "incident_id": "WELL-00019_34"
}
```

### `fleet_tool_call`

Emitted when the agent calls a tool during investigation.

```json
{
  "session_id": "a1b2c3d4",
  "well_id": "WELL-00019",
  "display_id": "Delta-19",
  "t": "2012-06-02T20:59:00",
  "call_id": "sys_get_window",
  "tool": "get_window",
  "args": {"minutes": 60},
  "requested_by": "system"
}
```

### `fleet_tool_result`

Emitted when a tool returns results.

```json
{
  "session_id": "a1b2c3d4",
  "well_id": "WELL-00019",
  "display_id": "Delta-19",
  "t": "2012-06-02T20:59:00",
  "call_id": "sys_get_window",
  "tool": "get_window",
  "result": {"available": true, "source": "replay_data", "...": "..."},
  "requested_by": "system"
}
```

### `fleet_assessment`

Emitted after agent investigation completes for a well.

```json
{
  "session_id": "a1b2c3d4",
  "well_id": "WELL-00019",
  "display_id": "Delta-19",
  "incident_id": "WELL-00019_34",
  "decision": "ALERT",
  "category": "review_now",
  "confidence": 0.82,
  "diagnosis": "hydrate_production_line",
  "brief": "Pressure and temperature both declining. 65% hydrate probability.",
  "next_action": "",
  "next_check_minute": null,
  "uncertainty": "",
  "evidence_refs": [
    {"tool": "classify_event", "summary": "p_hydrate=0.65"},
    {"tool": "hydrate_margin", "summary": "margin=-1.3C (danger)"}
  ],
  "playbook_refs": ["hydrate_response"],
  "source": "agent"
}
```

### `fleet_status`

Emitted after each tick and after investigations. Contains the global priority list.

```json
{
  "session_id": "a1b2c3d4",
  "held": false,
  "reason": "",
  "current_minute": 34,
  "priority_list": [
    {
      "well_id": "WELL-00019",
      "display_id": "Delta-19",
      "category": "review_now",
      "priority_rank": 1,
      "headline": "Hydrate forming detected. Recommend inhibitor injection.",
      "score": 0.82,
      "unacknowledged": true,
      "next_check_minute": null
    },
    {
      "well_id": "WELL-00006",
      "display_id": "Charlie-6",
      "category": "normal",
      "priority_rank": 2,
      "headline": "Normal operation",
      "score": 0.0,
      "unacknowledged": false,
      "next_check_minute": null
    }
  ],
  "counters": {
    "needs_review": 1,
    "unacknowledged": 1,
    "reliable_feeds": 4,
    "checks_due": 0
  }
}
```

### `fleet_end`

Emitted when all wells have finished or the session ends.

```json
{
  "session_id": "a1b2c3d4",
  "total_minutes": 180,
  "llm_attempts_used": 3
}
```

---

## Priority Categories

| Category | Condition | Color suggestion |
|----------|-----------|-----------------|
| `review_now` | score >= 0.7 or ALERT decision | Red |
| `investigate` | score >= 0.45 or WATCH decision | Orange |
| `watch` | score >= 0.2 | Yellow |
| `normal` | everything else | Green |

## Sort Order

Within each category, wells are sorted by:
1. Highest score first
2. Longest time unacknowledged
3. Alphabetical by well_id (stable tiebreaker)
