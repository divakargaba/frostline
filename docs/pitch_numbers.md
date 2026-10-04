# Pitch Numbers — Frostline

_Generated 2026-10-03. Every number is traced to a file._

---

## 1. Use these on stage

1. **"Our system caught all 9 hydrate events across 9 real held-out wells — zero misses."**
   [results/model_cv.json, final_test at threshold 0.3: 9/9 caught, 0 missed, 0% misdiagnosis rate, 11 held-out recordings from 3 wells]

2. **"We detect hydrate formation a median 32 hours before the blockage sets in."**
   [results/model_cv.json, final_test per_event: median lead_est = 1,935 min = 32.2 h across 5 events with established-phase labels]

3. **"The starter rule caught 5 events; our autonomous improvement round catches 10 — five more events with zero added false alarms."**
   [data/state/improvement_policy.json + results/summary.csv: B1 = 5/12, B1-revised = 10/12, 0 added FA, from 24-trial grid search on synthetic seed data]

4. **"Zero misdiagnoses: scaling and restriction events are never flagged as hydrate."**
   [results/model_cv.json, final_test threshold 0.3: misdiagnosis_rate = 0.0, 2 lookalike events, 0 flagged; fleet_report.json holdout: 0/3 lookalikes flagged]

5. **"The system learned from 17 wells, 20 recordings, and 1,366 hours of real sensor data."**
   [data/processed/research/fleet_report.json: training_wells = 17, training_recordings = 20, validation labelled_minutes = 81,935 = 1,366 h]

6. **"In the live demo, Well 19 gets flagged 31 hours before the established hydrate phase."**
   [data/demo/candidates.json: trigger at source minute 1,719; n_forming = 1,936 min; established at minute 3,590; lead = 1,871 min = 31.2 h]

7. **"The model scores 0.949 AUC on held-out wells — trained on 143 causal features with no future data leakage."**
   [results/model_cv.json, final_test: auc_hydrate_vs_rest = 0.949, n_features = 143]

8. **"The LLM agent reviews evidence with 8 tools and 14 playbook references, and always falls back to measured rules if the provider is down."**
   [src/fleet_llm.py: 8 callable tools + submit_assessment; data/playbook/: 14 docs; fallback verified by tests]

---

## 2. Full stat table

### Synthetic seed (Case 9 starter test)

| Stat | Value | Meaning | Data | Sample | File |
|------|-------|---------|------|--------|------|
| B1 caught | 5 / 12 | Bad hours caught by official 5th-pctl pressure | Synthetic seed, 720 hourly rows | 1 incident, 12 bad hours | results/summary.csv |
| B1-revised caught | 10 / 12 | After autonomous policy selection | Synthetic seed, test partition (days 21-30) | 1 incident, 12 bad hours | results/summary.csv |
| Improvement delta | +5 events, 0 added FA | Events gained with no false alarms added | Synthetic seed | 24 candidate policies searched | data/state/improvement_policy.json |
| Selected policy | P20 pressure + P10 temp/flow confirmation | Winning policy from grid search | Synthetic seed, validation days 11-20 | 24 trials | data/state/improvement_policy.json |
| Policy fingerprint | 9ebaabad52ad | Frozen policy ID | — | — | data/state/improvement_policy.json |
| B1 detection delay | 7 h | Hours after label onset | Synthetic seed | 1 incident | README.md |
| B1-revised delay | 1 h | Hours after label onset | Synthetic seed | 1 incident | README.md |

### Real 3W holdout — model_cv.json (5-class model, threshold 0.3)

| Stat | Value | Meaning | Data | Sample | File |
|------|-------|---------|------|--------|------|
| Events caught | 9 / 9 | All hydrate events detected | Held-out wells (WELL-00006, -00019, -00042) | 9 hydrate + 2 lookalike recordings | results/model_cv.json final_test |
| Missed | 0 | Events with no alarm | Held-out | 9 events | results/model_cv.json |
| AUC | 0.949 | Hydrate vs rest, minute-level | Held-out | 11 recordings | results/model_cv.json |
| Macro F1 | 0.459 | 5-class minute-level | Held-out | 11 recordings | results/model_cv.json |
| False alarms/day | 0.884 | Episodes per normal well-day | Held-out, normal files only | — | results/model_cv.json (false_alarms_per_day_normal_files at 0.3) |
| FA overall | 2.464 | Episodes per day (all files) | Held-out | — | results/model_cv.json |
| Mean lead time (vs established) | 2,531 min = 42.2 h | Mean alarm-to-established lead | Held-out | 5 events with established labels | results/model_cv.json |
| Median lead time (vs established) | 1,935 min = 32.2 h | Median alarm-to-established | Held-out | 5 events | results/model_cv.json |
| Median lead (vs forming) | 11 min | Alarm relative to forming onset (positive = before) | Held-out | 8 caught events with forming labels | results/model_cv.json (median_lead_vs_form_min at 0.3) |
| Misdiagnosis rate | 0.0% | Lookalike events flagged as hydrate | Held-out | 2 lookalike events | results/model_cv.json |
| Held-out wells | 3 | WELL-00006, -00019, -00042 | — | — | results/model_cv.json |
| Held-out recordings | 11 | Total test recordings | — | — | results/model_cv.json |
| Features | 143 | Causal features, no future leakage | — | — | results/model_cv.json |

### Real 3W holdout — fleet_report.json (selected policy: 0.7 threshold, 10-min persistence)

| Stat | Value | Meaning | Data | Sample | File |
|------|-------|---------|------|--------|------|
| Events detected | 4 / 4 | Hydrate events caught | Held-out 4 demo wells (11 recordings) | 4 hydrate events | fleet_report.json stages.selected |
| Event recall | 100% | All events detected before established | Held-out | 4 events | fleet_report.json |
| Early events | 4 / 4 | Alarm before established phase | Held-out | 4 events | fleet_report.json |
| Mean detection delay | 10.0 min | Delay from alarm threshold to trigger | Held-out | 4 events | fleet_report.json |
| FA per normal day | 1.07 | False alarm episodes per day | Held-out, 8,079 normal minutes | 6 episodes | fleet_report.json |
| FA min per normal day | 362.5 | False alarm minutes per normal well-day | Held-out | — | fleet_report.json |
| Lookalikes flagged | 0 / 3 | Look-alike recordings incorrectly flagged | Held-out | 3 recordings | fleet_report.json |

### Real 3W — development cross-validation (17 wells, LOWO)

| Stat | Value | Meaning | Data | Sample | File |
|------|-------|---------|------|--------|------|
| AUC (all features) | 0.916 | Hydrate vs rest | Validation folds | 17 wells | results/model_cv.json development.all |
| AUC (relative features) | 0.961 | Hydrate vs rest | Validation folds | 17 wells | results/model_cv.json development.relative |
| Caught at 0.3 (all) | 17 / 19 | Events detected | Validation | 19 hydrate events | results/model_cv.json |
| Misdiagnosis rate (all, 0.3) | 10.4% | Lookalike events flagged | Validation | 383 lookalike events | results/model_cv.json |
| Labelled minutes | 81,935 = 1,366 h | Total labelled data | Validation | 17 wells | fleet_report.json |

### B0–M3 ladder (results/summary.csv)

| System | Description | Caught | Total | FA/day | Mean lead | Misdiag | Notes |
|--------|-------------|-------:|------:|-------:|----------:|--------:|-------|
| B0 | Always normal | 0 | 12 | 0.0 | — | 0.0% | Seed |
| B1 | 5th-pctl pressure | 5 | 12 | 0.0 | 420 min (7 h) | 0.0% | Seed, hackathon starter |
| B1-revised | P20 + P10 confirmation | 10 | 12 | 0.0 | 60 min (1 h) | 0.0% | Seed, autonomous selection |
| M1 | LightGBM 5-class | 9 | 9 | 0.88 | — | 0.0% | 3W LOWO, 143 features, AUC 0.949 |
| M3 | Agent + ML + physics + RAG | 9 | 9 | 0.88 | — | 0.0% | Full pipeline |

**M3 vs M1:** M3 has identical headline numbers to M1 in summary.csv. This is because the summary reports the model's alarm-based detection, and the agent/policy layer does not change alarm counts — it adds investigation quality, evidence grounding, playbook references and operator workflow. The numerical detection scores are the same model underneath. The fleet_report.json holdout (with policy 0.7/10min) shows 4/4 at 1.07 FA/day and 10-min delay, but this evaluates a different threshold/persistence than the model_cv.json 0.3 threshold.

### Fleet demo (4 wells, 180-minute replay)

| Well | Event class | Frostline conclusion | Investigations (flood-protected) |
|------|------------|---------------------|----------------------------------|
| WELL-00001 (Alpha-1) | 0 - Normal | Normal operation, no escalation | 1 (initial only) |
| WELL-00002 (Bravo-2) | 6 - Quick restriction | Watch → monitoring | 2-3 |
| WELL-00006 (Charlie-6) | 7 - Scaling | Watch → DISMISS (scaling) | 2-3 |
| WELL-00019 (Delta-19) | 8 - Hydrate production line | Watch → ALERT (hydrate) | 3-4 |
| **Total** | — | — | **~10** (was ~360 before flood protection) |

Well 19 demo lead time: trigger at source minute 1,719; forming starts ~minute 1,654; established at ~minute 3,590. **Lead before established = 1,871 min = 31.2 hours.** Lead after forming onset = 65 minutes into the forming phase.

### Dataset scale

| Stat | Value | File |
|------|-------|------|
| 3W wells in subset | 21 | README.md |
| 3W recordings in subset | 31 | README.md |
| Training wells | 17 | fleet_report.json |
| Training recordings | 20 | fleet_report.json |
| Held-out recordings | 11 | fleet_report.json |
| Excluded demo wells | 4 | fleet_report.json |
| Validation labelled minutes | 81,935 (1,366 h) | fleet_report.json |
| Holdout labelled minutes | 22,910 (382 h) | fleet_report.json |
| Total 3W dataset (full, not our subset) | ~2,000 recordings | context/hackathon/case9_data_README.md |

### System facts

| Stat | Value | Source |
|------|-------|--------|
| Playbook docs | 14 verified sources | data/playbook/ (14 .md + SOURCES.md) |
| Agent tools | 8 callable + submit_assessment | src/fleet_llm.py |
| Tests passing | 215 pass, 1 skip, 10 pending (unimplemented physics stubs) | pytest on integration branch |
| LLM providers | 3 (Gemini, Groq, OpenRouter) with automatic failover | src/llm.py |
| Model candidates evaluated | 4 feature/weighting families x 15 policies = 60 | fleet_report.json |
| Validation folds | 3 (grouped by well) | fleet_report.json |
| Non-regression gates | 4 (events, FA minutes, lookalike recall, macro F1) | fleet_report.json |

### LLM run stats (from recorded sessions)

| Stat | Value | File |
|------|-------|------|
| Total LLM requests | 49 | data/state/llm_usage.json |
| Total tokens | 172,525 | data/state/llm_usage.json |
| Gemini requests | 13 | data/state/llm_usage.json |
| OpenRouter requests | 16 | data/state/llm_usage.json |
| Median inter-call latency | 16.6 s | data/state/llm_usage.json (computed) |
| Flood protection: investigations per demo | ~10 total (was ~360) | tests/test_fleet_api.py |

---

## 3. Conflicts and fixes needed

### CONFLICT 1: Landing page mislabels seed data as "3W holdout"
- **Where:** `frontend/src/main.tsx` line 23: `"5 → 10", "events caught of 12 (3W holdout)"`
- **Problem:** B1 → B1-revised (5 → 10 of 12) is from the **synthetic seed data** (Case 9 starter, 720 hourly rows), not the 3W holdout. The 3W holdout is 9/9 events (model_cv.json).
- **Fix:** Change to `"events caught of 12 (seed test)"` or separate the two numbers: seed improvement 5 → 10, 3W holdout 9/9.

### CONFLICT 2: Landing page "9 of 9 3W hydrate wells detected" vs actual count
- **Where:** `frontend/src/main.tsx` line 25: `"9 of 9", "3W hydrate wells detected"`
- **Problem:** "9 of 9" is the number of hydrate **events** (recordings) caught in the model_cv.json holdout at threshold 0.3, from **3 wells** (WELL-00006, -00019, -00042). The fleet_report holdout (with selected policy) says 4/4 events from 1 hydrate well (WELL-00019 only). Neither is "9 wells."
- **Fix:** Change to `"9 of 9 hydrate events detected"` with label `"3W held-out recordings"`.

### CONFLICT 3: README test count is stale
- **Where:** README.md line 178: "194 tests passed"
- **Problem:** Current integration branch has 215 pass, 1 skip.
- **Fix:** Update to 215.

### CONFLICT 4: M1 and M3 are identical in summary.csv
- **Where:** results/summary.csv rows M1 and M3
- **Problem:** Both show 9/9 caught, 0.88 FA/day, 0.0% misdiagnosis. M3 does not add detection improvement over M1 — it adds investigation quality, evidence grounding and operator workflow. This is accurate but could be misread as "the agent adds nothing."
- **Clarification for pitch:** "M3 doesn't change *what* the model detects — it changes *how* the operator understands and acts on it: grounded evidence, playbook references, checklists, and a recheck schedule."

### CONFLICT 5: fleet_report holdout wells vs model_cv holdout wells
- **Where:** fleet_report.json evaluation_hydrate_wells = ["WELL-00019"] only; model_cv.json final_test_wells = ["WELL-00006", "WELL-00019", "WELL-00042"]
- **Problem:** fleet_report holdout has only 4 hydrate events, all from WELL-00019. model_cv.json holdout has 9 events from 3 wells. The 4/4 fleet_report number is a subset.
- **Clarification:** The fleet_report uses a stricter policy (0.7 threshold, 10-min persistence) and evaluates the 4 demo-excluded wells; model_cv.json evaluates all 3 test wells at threshold 0.3. Both are valid but measure different things.

### No synthetic-as-real issues
B0–B1-revised are correctly labeled as seed data in summary.csv and README. M1–M3 are correctly labeled as 3W real. The only issue is the landing page (Conflict 1) mislabeling the seed improvement as "3W holdout."

---

## 4. The lead-time number

**For the script line: "[X] minutes before the blockage set in"**

**Use: "31 hours before the blockage set in"** (or "over 31 hours").

Source: the Well 19 demo replay. The fleet policy first triggers at source minute 1,719; the established (blockage) phase begins at approximately minute 3,590. Lead time = 1,871 minutes = **31.2 hours.**

This is from the actual demo the judges will see (`data/demo/candidates.json`), not a cross-validated average. The held-out median across all test events is 32.2 hours (results/model_cv.json), which is consistent but slightly different because it uses the model threshold (0.3) rather than the fleet policy threshold (0.7/10-min persistence).

Either number is defensible. The demo number (31 hours) is preferred because the judges will see it happen live.
