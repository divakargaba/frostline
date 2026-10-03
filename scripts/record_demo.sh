#!/usr/bin/env bash
# Record a fleet demo using the running backend (Mico's fleet API).
#
# Prerequisites:
#   - Backend running on port 8000: uvicorn backend.main:app --port 8000
#   - Fleet model trained: python scripts/train_fleet_model.py
#   - 3W data fetched: bash scripts/fetch_3w.sh
#
# Usage:
#   bash scripts/record_demo.sh            # rules-only (no LLM)
#   USE_LLM=1 bash scripts/record_demo.sh  # with live LLM (needs API key + quota)
#
# Output: data/demo/fleet-recording-<id>.json

set -euo pipefail

BASE="http://127.0.0.1:8000/api/fleet"
OUTDIR="data/demo"
mkdir -p "$OUTDIR"

echo "=== Frostline Fleet Demo Recorder ==="
echo ""

# 1. Health check
echo "Checking backend health..."
HEALTH=$(curl -sf "$BASE/capabilities" 2>/dev/null || echo '{"model_ready":false}')
MODEL_READY=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model_ready', False))")
if [ "$MODEL_READY" != "True" ]; then
    echo "ERROR: Fleet model not ready. Run: python scripts/train_fleet_model.py"
    exit 1
fi
echo "Fleet model ready."

# 2. Create session (guided mode, speed=120 for fast replay)
echo "Creating fleet session..."
SESSION=$(curl -sf -X POST "$BASE/sessions" \
    -H "Content-Type: application/json" \
    -d '{"speed": 120, "mode": "continuous"}')
SESSION_ID=$(echo "$SESSION" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "Session ID: $SESSION_ID"

# 3. Start replay
echo "Starting replay..."
curl -sf -X POST "$BASE/sessions/$SESSION_ID/control" \
    -H "Content-Type: application/json" \
    -d '{"action": "start"}' > /dev/null

# 4. Wait for completion (poll every 5s)
echo "Waiting for replay to finish..."
while true; do
    SNAP=$(curl -sf "$BASE/sessions/$SESSION_ID")
    STATUS=$(echo "$SNAP" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
    INDEX=$(echo "$SNAP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['index']}/{d['total']}\")")
    echo "  Status: $STATUS, Progress: $INDEX"
    if [ "$STATUS" = "completed" ] || [ "$STATUS" = "ended" ] || [ "$STATUS" = "error" ]; then
        break
    fi
    sleep 5
done

# 5. Export
OUTFILE="$OUTDIR/fleet-recording-${SESSION_ID:0:8}.json"
echo "Exporting to $OUTFILE..."
curl -sf "$BASE/sessions/$SESSION_ID/export" -o "$OUTFILE"

# 6. Summary
WELLS=$(python3 -c "
import json
data = json.load(open('$OUTFILE'))
for w in data['run']['wells']:
    print(f\"  {w['well_id']}: status={w['status']}, investigation={w['investigation']}\")
")
echo ""
echo "=== Recording complete ==="
echo "$WELLS"
echo ""
echo "File: $OUTFILE"
echo "Size: $(du -h "$OUTFILE" | cut -f1)"
