#!/usr/bin/env bash
# Orchestrate v03 SN-search (Steps 5-6 — Steps 1-3 run separately).
set -u
PY=/opt/miniconda3/bin/python
DIR=/Users/suzuki/github/projects_cosmos/programs_webpage
LOG=/Users/suzuki/github/projects_cosmos/csvfiles_sn/run_v03_status.log
FAIL=/Users/suzuki/github/projects_cosmos/csvfiles_sn/run_v03_FAILURE.txt
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] PIPELINE: $*" >> "$LOG"; }
fail() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL in $1: $2" >> "$LOG"; echo "FAIL in $1: $2" > "$FAIL"; }

log "=== START v03 pipeline (Steps 5 + 6 — assumes 1-3 already done) ==="
T0=$(date +%s)

log "--- Step 5 v03: train_cnn_v03.py ---"
if ! "$PY" "$DIR/train_cnn_v03.py" >> "$LOG" 2>&1; then
    fail "step5_v03" "exited non-zero"; exit 5; fi
T5=$(date +%s); log "Step 5 v03 wallclock: $((T5-T0))s"

log "--- Step 6 v03: infer_sn_v03.py ---"
if ! "$PY" "$DIR/infer_sn_v03.py" >> "$LOG" 2>&1; then
    fail "step6_v03" "exited non-zero"; exit 6; fi
T6=$(date +%s); log "Step 6 v03 wallclock: $((T6-T5))s"

log "=== DONE. Total: $((T6-T0))s ==="
ls -la /Users/suzuki/github/projects_cosmos/csvfiles_sn/cnn_models_v03.pt \
       /Users/suzuki/github/projects_cosmos/csvfiles_sn/training_summary_v03.txt \
       /Users/suzuki/github/projects_cosmos/csvfiles_sn/tbl_sn_candidates_v03.csv \
       >> "$LOG" 2>&1
