#!/bin/bash
# Auto-commit + push htmls/sn_search/v03 every 5 minutes while inference runs.
# Exits cleanly once the main inference and VIS inference processes are gone.
#
# Usage: bash auto_commit_v03.sh &
# Stop manually with `kill <pid>` (logs go to /tmp/auto_commit_v03.log).

set -u
REPO=/Users/suzuki/github/projects_cosmos
LOG=/tmp/auto_commit_v03.log
INTERVAL=300   # seconds = 5 min
TARGET_DIR=htmls/sn_search/v03

# Walk-up: capture the inference PIDs that exist NOW so the loop stops when
# they vanish. If you start more later, restart this script.
INFER_PIDS=$(pgrep -f "infer_sn_v03.py|add_vis_v03.py" | tr '\n' ' ')
echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto-commit started, watching PIDs: $INFER_PIDS" | tee -a "$LOG"

cd "$REPO"

commit_n=0
while true; do
    # Stage current state of the v03 webpage tree (PNGs, HTML, gitignore).
    git add -- "$TARGET_DIR" 2>>"$LOG"

    if git diff --cached --quiet; then
        echo "[$(date '+%H:%M:%S')] no changes" | tee -a "$LOG"
    else
        commit_n=$((commit_n + 1))
        # Quick stats for the commit body
        n_cand=$(awk 'END{print NR-1}' /Users/suzuki/github/projects_cosmos/csvfiles_sn/tbl_sn_candidates_v03.csv 2>/dev/null || echo "?")
        n_png=$(ls "$TARGET_DIR"/polished_top100/polished_*_hst.png 2>/dev/null | wc -l | tr -d ' ')
        msg="v03 SN search: streaming update #${commit_n} ($(date '+%H:%M:%S'))"
        body="Candidates in CSV: ${n_cand} ; rendered candidates on disk: ${n_png}"
        git commit -m "$msg" -m "$body" >>"$LOG" 2>&1 \
            && echo "[$(date '+%H:%M:%S')] committed: $msg ($body)" | tee -a "$LOG"
        git push >>"$LOG" 2>&1 \
            && echo "[$(date '+%H:%M:%S')] pushed" | tee -a "$LOG"
    fi

    # Stop when ALL the originally-tracked inference processes are gone.
    alive=0
    for p in $INFER_PIDS; do
        if ps -p "$p" >/dev/null 2>&1; then alive=1; break; fi
    done
    if [ "$alive" = "0" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] inference pids all gone; doing FINAL commit + push then exit" | tee -a "$LOG"
        # Do one more pass in case there's a tail of new files
        git add -- "$TARGET_DIR" 2>>"$LOG"
        if ! git diff --cached --quiet; then
            commit_n=$((commit_n + 1))
            n_cand=$(awk 'END{print NR-1}' /Users/suzuki/github/projects_cosmos/csvfiles_sn/tbl_sn_candidates_v03.csv 2>/dev/null || echo "?")
            n_png=$(ls "$TARGET_DIR"/polished_top100/polished_*_hst.png 2>/dev/null | wc -l | tr -d ' ')
            git commit -m "v03 SN search: FINAL update #${commit_n}" \
                       -m "Inference complete. Candidates: ${n_cand}, rendered: ${n_png}" >>"$LOG" 2>&1
            git push >>"$LOG" 2>&1
            echo "[$(date '+%H:%M:%S')] FINAL committed + pushed" | tee -a "$LOG"
        fi
        exit 0
    fi
    sleep "$INTERVAL"
done
