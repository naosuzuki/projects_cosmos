#!/bin/bash
# 74_run_piff_all.sh — generate Piff panels (73_) for every tile-suffix the
# QA site serves from OUR data, then refresh the site + deploy repo.
# Serial + taskpolicy-throttled (machine I/O rule: one pipeline at a time).
# Resumable: skips any tile whose psf_residuals_piff.png already exists.
# Euclid + HSC are EXCLUDED — their pages are imported from the hostgalxy
# record, so Piff panels in our tree would not attach to the toggle.
set -u
cd /Users/suzuki/github/projects_cosmos/programs_star/v04
# Cap the Piff fit's BLAS/OpenMP fan-out to 2 cores (of 14) so a
# concurrently-running SExtractor keeps the rest.  The PixelGrid lstsq
# otherwise spins up one thread per core and saturates the CPU.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/opt/miniconda3/bin/python
W=/Volumes/exdisk1/data/photometry_v04
LOG=/tmp/piff; mkdir -p $LOG
: > $LOG/loop.log

refresh_site() {
  if mkdir /tmp/piffsite.lock 2>/dev/null; then
    taskpolicy -c utility $PY 55_make_psf_qa_site.py >/dev/null 2>&1
    rmdir /tmp/piffsite.lock
  fi
}

run_to() {                      # run_to <secs> <cmd...>  (macOS-safe timeout)
  local secs=$1; shift
  "$@" & local pid=$!
  ( sleep "$secs"; kill -9 $pid 2>/dev/null ) & local k=$!
  wait $pid 2>/dev/null; local rc=$?
  kill $k 2>/dev/null; wait $k 2>/dev/null
  return $rc
}

# ── build the worklist in priority order (space first; SDSS last) ──
: > $LOG/work.txt
for pref in hst_acs_f814w jwst_nircam_ unwise_ ps1_ lsdr10_ sdss_; do
  for psf in $W/${pref}*/*/psf/stars_*.psf; do
    [ -e "$psf" ] || continue
    d=$(dirname "$psf"); inst=$(basename $(dirname "$d")); tile=$(basename $(dirname "$(dirname "$d")"))
    # robust part extraction
    inst=$(echo "$psf" | sed -E "s#$W/([^/]+)/([^/]+)/psf/.*#\1#")
    tile=$(echo "$psf" | sed -E "s#$W/([^/]+)/([^/]+)/psf/.*#\2#")
    suffix=$(basename "$psf"); suffix=${suffix#stars_}; suffix=${suffix%.psf}
    [ -e "$W/$inst/$tile/psf/psf_${suffix}.meta.json" ] || continue
    [ -e "$W/$inst/$tile/psf/psf_residuals_piff.png" ] && continue   # resume-skip
    echo "$inst $tile $suffix" >> $LOG/work.txt
  done
done
N=$(wc -l < $LOG/work.txt | tr -d ' ')
echo "worklist: $N tile-suffixes" >> $LOG/loop.log

i=0
while read -r inst tile suffix; do
  i=$((i+1))
  run_to 900 taskpolicy -c utility $PY 73_make_piff_panels.py \
      --instrument "$inst" --tile "$tile" --suffix "$suffix" \
      >>$LOG/last.txt 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -e "$W/$inst/$tile/psf/psf_residuals_piff.png" ]; then
    echo "OK   $i/$N $inst $tile $suffix" >> $LOG/loop.log
  elif [ $rc -eq 0 ]; then
    echo "PART $i/$N $inst $tile $suffix (chi2 only, no residual)" >> $LOG/loop.log
  else
    echo "FAIL $i/$N $inst $tile $suffix rc=$rc" >> $LOG/loop.log
  fi
  [ $((i % 25)) -eq 0 ] && refresh_site
done < $LOG/work.txt

refresh_site
# deploy repo export + commit (inside this detached process → no permission prompt)
taskpolicy -c utility $PY 57_export_psf_qa_site.py --jobs 4 >>$LOG/export.log 2>&1
cd /Users/suzuki/github/cosmos_psf_qa_site
git add -A
git -c user.name="Nao Suzuki" -c user.email="nao.suzuki@gmail.com" \
    commit -q -m "Piff panels + PSFEx/Piff toggle across all served surveys (74_ batch)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" 2>>$LOG/git.log
git push -q 2>>$LOG/git.log
echo "PIFF ALL COMPLETE $(date '+%Y-%m-%d %H:%M')  ok=$(grep -c '^OK' $LOG/loop.log) fail=$(grep -c '^FAIL' $LOG/loop.log) part=$(grep -c '^PART' $LOG/loop.log)" >> $LOG/loop.log
