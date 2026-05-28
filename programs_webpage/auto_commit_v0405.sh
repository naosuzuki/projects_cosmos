#!/bin/bash
# Auto-commit + push the htmls/sn_search/ HTML site (v04 + v05) every 5
# minutes while either pipeline is still running.
set -u
REPO=/Users/suzuki/github/projects_cosmos
LOG=/tmp/auto_commit_v0405.log
INTERVAL=300
: > "$LOG"
echo "[$(date '+%H:%M:%S')] auto_commit_v0405 starting (stages all of htmls/sn_search/)" | tee -a "$LOG"
cd "$REPO"
n=0
while true; do
    # Stage the entire htmls/sn_search/ tree (covers v04, v05, and any subdirs)
    git add -A -- htmls/sn_search/ 2>>"$LOG" || true
    if ! git diff --cached --quiet; then
        n=$((n+1))
        nv4=$(awk 'END{print NR-1}' csvfiles_sn/tbl_sn_candidates_v04.csv 2>/dev/null || echo "?")
        nv5=$(awk 'END{print NR-1}' csvfiles_sn/tbl_sn_candidates_v05.csv 2>/dev/null || echo "?")
        msg="v04/v05 streaming update #${n} ($(date '+%H:%M:%S'))"
        body="v04 candidates: ${nv4} ; v05 candidates: ${nv5}"
        git commit -m "$msg" -m "$body" >>"$LOG" 2>&1 && echo "[$(date '+%H:%M:%S')] committed: $body" | tee -a "$LOG"
        git push >>"$LOG" 2>&1 && echo "[$(date '+%H:%M:%S')] pushed" | tee -a "$LOG"
    else
        echo "[$(date '+%H:%M:%S')] no changes" | tee -a "$LOG"
    fi
    # Exit when both pipelines are gone
    alive_v04=0; alive_v05=0
    pgrep -f "infer_v04_diff.py|consolidate_v04.py|run_v04_consolidate" >/dev/null 2>&1 && alive_v04=1
    pgrep -f "make_training_set_v05.py|train_cnn_v05.py|infer_v05.py|consolidate_v05.py|run_v05_chain" >/dev/null 2>&1 && alive_v05=1
    if [ "$alive_v04" = "0" ] && [ "$alive_v05" = "0" ]; then
        echo "[$(date '+%H:%M:%S')] both pipelines done; FINAL commit" | tee -a "$LOG"
        git add -A -- htmls/sn_search/ 2>>"$LOG" || true
        if ! git diff --cached --quiet; then
            n=$((n+1))
            git commit -m "v04/v05 FINAL update #${n}" >>"$LOG" 2>&1
            git push >>"$LOG" 2>&1
        fi
        exit 0
    fi
    sleep "$INTERVAL"
done
