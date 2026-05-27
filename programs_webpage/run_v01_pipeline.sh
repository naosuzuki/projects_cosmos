#!/usr/bin/env bash
# Orchestrate Steps 3-6 of v01 SN hunt. Run in background; log to status file.
set -u   # don't `set -e` — we want to log failures per phase and continue if possible
PY=/opt/miniconda3/bin/python
DIR=/Users/suzuki/github/projects_cosmos/programs_webpage
LOG=/Users/suzuki/github/projects_cosmos/csvfiles_sn/run_v01_status.log
FAIL=/Users/suzuki/github/projects_cosmos/csvfiles_sn/run_v01_FAILURE.txt

mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] PIPELINE: $*" | tee -a "$LOG"; }
fail() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILURE in $1: $2" | tee -a "$LOG" | tee -a "$FAIL"; }

log "=== START v01 pipeline ==="
T0=$(date +%s)

# Step 3
log "--- Step 3: make_training_set_v01.py ---"
if ! "$PY" "$DIR/make_training_set_v01.py" 2>&1 | tee -a "$LOG"; then
    fail "step3" "make_training_set_v01.py exited non-zero"
    exit 3
fi
T3=$(date +%s); log "Step 3 wallclock: $((T3-T0))s"

# Step 5 (step 4 is just code definition, nothing to run)
log "--- Step 5: train_cnn_v01.py ---"
if ! "$PY" "$DIR/train_cnn_v01.py" 2>&1 | tee -a "$LOG"; then
    fail "step5" "train_cnn_v01.py exited non-zero"
    exit 5
fi
T5=$(date +%s); log "Step 5 wallclock: $((T5-T3))s"

# Step 6
log "--- Step 6: infer_sn_v01.py ---"
if ! "$PY" "$DIR/infer_sn_v01.py" 2>&1 | tee -a "$LOG"; then
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
       /Users/suzuki/github/projects_cosmos/csvfiles_sn/sn_candidates_v01_all_scored.parquet 2>&1 | tee -a "$LOG"
ls -la /Users/suzuki/github/projects_cosmos/htmls/sn_search/v01/ 2>&1 | tee -a "$LOG"
