#!/bin/bash
# Re-consolidate v04 PIXEL-diff every 3 min while diff imaging runs.
set -u
PY=/opt/miniconda3/bin/python
DIR=/Users/suzuki/github/projects_cosmos/programs_webpage
LOG=/tmp/v04_consolidate_loop.log
: > "$LOG"
echo "[$(date '+%H:%M:%S')] v04 pixel-diff consolidate loop starting" | tee -a "$LOG"
while pgrep -f "infer_v04_diff.py" >/dev/null 2>&1; do
    "$PY" "$DIR/consolidate_v04.py" >>"$LOG" 2>&1 || true
    sleep 180
done
"$PY" "$DIR/consolidate_v04.py" >>"$LOG" 2>&1 || true
echo "[$(date '+%H:%M:%S')] v04 consolidate loop exiting" | tee -a "$LOG"
