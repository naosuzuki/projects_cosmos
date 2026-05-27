#!/usr/bin/env bash
# Orchestrate v02 (re-train + re-infer using corrected per-survey labels).
# Step 3 is skipped — reuses training_set_v01.npz.
set -u
PY=/opt/miniconda3/bin/python
DIR=/Users/suzuki/github/projects_cosmos/programs_webpage
LOG=/Users/suzuki/github/projects_cosmos/csvfiles_sn/run_v02_status.log
FAIL=/Users/suzuki/github/projects_cosmos/csvfiles_sn/run_v02_FAILURE.txt

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] PIPELINE: $*" >> "$LOG"; }
fail() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILURE in $1: $2" >> "$LOG"; echo "FAILURE in $1: $2" > "$FAIL"; }

log "=== START v02 pipeline ==="
T0=$(date +%s)

log "--- Step 5 v02: train_cnn_v02.py ---"
if ! "$PY" "$DIR/train_cnn_v02.py" >> "$LOG" 2>&1; then
    fail "step5_v02" "train_cnn_v02.py exited non-zero"
    exit 5
fi
T5=$(date +%s); log "Step 5 v02 wallclock: $((T5-T0))s"

log "--- Step 6 v02: infer_sn_v02.py ---"
if ! "$PY" "$DIR/infer_sn_v02.py" >> "$LOG" 2>&1; then
    fail "step6_v02" "infer_sn_v02.py exited non-zero"
    exit 6
fi
T6=$(date +%s); log "Step 6 v02 wallclock: $((T6-T5))s"

log "=== DONE. Total: $((T6-T0))s ==="
log "Artifacts:"
ls -la /Users/suzuki/github/projects_cosmos/csvfiles_sn/cnn_models_v02.pt \
       /Users/suzuki/github/projects_cosmos/csvfiles_sn/training_summary_v02.txt \
       /Users/suzuki/github/projects_cosmos/csvfiles_sn/tbl_sn_candidates_v02.csv \
       >> "$LOG" 2>&1
ls -la /Users/suzuki/github/projects_cosmos/htmls/sn_search/v02/ >> "$LOG" 2>&1
