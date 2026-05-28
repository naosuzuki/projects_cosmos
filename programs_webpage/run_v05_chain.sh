#!/bin/bash
# Chain: wait for v05 training set, then train CNN, run inference, consolidate.
set -u
PY=/opt/miniconda3/bin/python
DIR=/Users/suzuki/github/projects_cosmos/programs_webpage
LOG=/tmp/v05_chain.log
NPZ=/Users/suzuki/github/projects_cosmos/csvfiles_sn/training_set_v05.npz
: > "$LOG"
echo "[$(date '+%H:%M:%S')] waiting for training_set_v05.npz ..." | tee -a "$LOG"
# Wait until make_training_set_v05.py finishes (its background ID is bmrt6ggki)
until pgrep -f "make_training_set_v05.py" >/dev/null 2>&1; do sleep 2; done
# Now poll for completion (file exists AND no make_training_set process running)
until [ -f "$NPZ" ] && ! pgrep -f "make_training_set_v05.py" >/dev/null 2>&1; do sleep 5; done
echo "[$(date '+%H:%M:%S')] training_set_v05.npz ready, starting train_cnn_v05.py" | tee -a "$LOG"
"$PY" "$DIR/train_cnn_v05.py" >>"$LOG" 2>&1 || { echo "train failed" | tee -a "$LOG"; exit 5; }
echo "[$(date '+%H:%M:%S')] training done, starting infer_v05.py" | tee -a "$LOG"
"$PY" "$DIR/infer_v05.py" >>"$LOG" 2>&1 || { echo "infer failed" | tee -a "$LOG"; exit 6; }
echo "[$(date '+%H:%M:%S')] inference done, consolidating" | tee -a "$LOG"
"$PY" "$DIR/consolidate_v05.py" >>"$LOG" 2>&1
echo "[$(date '+%H:%M:%S')] === v05 chain DONE ===" | tee -a "$LOG"
