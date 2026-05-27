#!/usr/bin/env bash
# Orchestrate Steps 3-6 of v01 SN hunt (parallel/MPS-optimised).
# Scripts print to stdout; orchestrator redirects all output to LOG (no tee → no dup).
set -u
PY=/opt/miniconda3/bin/python
DIR=/Users/suzuki/github/projects_cosmos/programs_webpage
LOG=/Users/suzuki/github/projects_cosmos/csvfiles_sn/run_v01_status.log
FAIL=/Users/suzuki/github/projects_cosmos/csvfiles_sn/run_v01_FAILURE.txt

mkdir -p "$(dirname "$LOG")"
: > "$LOG"   # truncate previous run's log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] PIPELINE: $*" >> "$LOG"; }
fail() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILURE in $1: $2" >> "$LOG"; echo "FAILURE in $1: $2" > "$FAIL"; }

log "=== START v01 pipeline (parallel) ==="
T0=$(date +%s)

log "--- Step 3: make_training_set_v01.py ---"
if ! "$PY" "$DIR/make_training_set_v01.py" >> "$LOG" 2>&1; then
    fail "step3" "make_training_set_v01.py exited non-zero"
    exit 3
fi
T3=$(date +%s); log "Step 3 wallclock: $((T3-T0))s"

log "--- Step 5: train_cnn_v01.py ---"
if ! "$PY" "$DIR/train_cnn_v01.py" >> "$LOG" 2>&1; then
    fail "step5" "train_cnn_v01.py exited non-zero"
    exit 5
fi
T5=$(date +%s); log "Step 5 wallclock: $((T5-T3))s"

log "--- Step 6: infer_sn_v01.py ---"
if ! "$PY" "$DIR/infer_sn_v01.py" >> "$LOG" 2>&1; then
    fail "step6" "infer_sn_v01.py exited non-zero"
    exit 6
fi
T6=$(date +%s); log "Step 6 wallclock: $((T6-T5))s"

log "=== DONE. Total wallclock: $((T6-T0))s ==="
log "Artifacts:"
ls -la /Users/suzuki/github/projects_cosmos/csvfiles_sn/training_set_v01.npz \
       /Users/suzuki/github/projects_cosmos/csvfiles_sn/cnn_models_v01.pt \
       /Users/suzuki/github/projects_cosmos/csvfiles_sn/training_summary_v01.txt \
       /Users/suzuki/github/projects_cosmos/csvfiles_sn/tbl_sn_candidates_v01.csv \
       /Users/suzuki/github/projects_cosmos/csvfiles_sn/sn_candidates_v01_all_scored.parquet \
       >> "$LOG" 2>&1
ls -la /Users/suzuki/github/projects_cosmos/htmls/sn_search/v01/ >> "$LOG" 2>&1
