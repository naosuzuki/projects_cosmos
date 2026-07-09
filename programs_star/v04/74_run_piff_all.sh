#!/bin/bash
# 74_run_piff_all.sh — generate Piff panels (73_) for every tile-suffix the
# QA site serves from OUR data + the HSC record, then refresh the site.
# Serial + taskpolicy-throttled (machine I/O rule: one pipeline at a time).
# Resumable: skips any tile whose psf_residuals_piff.png already exists.
# HSC uses the hostgalxy record (psf_v01, SSD) via 73_ --flat --root; its
# source coadds are 1.8 GB each on My Book, so it MUST stay in this one
# serial pipeline (never concurrent).  Euclid stays excluded (imported).
set -u
cd /Users/suzuki/github/projects_cosmos/programs_star/v04
# CPU/RAM are plentiful (73% idle, 18 GB free) and the concurrent hostgalxy
# SExtractor stages to the internal SSD, so it does NOT contend for the My
# Book USB disk where our source FITS live.  The only per-tile cost is disk
# read latency on that USB drive (worse for the larger LS DR10 images).  So
# leave BLAS/OpenMP UNCAPPED — full cores make each compute burst finish
# fast, clearing the disk-read/compute cycle sooner.
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

# ── build the worklist.  Columns: inst tile suffix [flat] [root]
#    HSC record first (fast, on SSD), then served surveys space→SDSS. ──
: > $LOG/work.txt
HSCROOT=/Users/suzuki/data/psf_v01
for inst in hsc_g hsc_r2 hsc_i2 hsc_z hsc_y; do
  for meta in $HSCROOT/$inst/*/psf/psf.meta.json; do
    [ -e "$meta" ] || continue
    tract=$(echo "$meta" | sed -E "s#$HSCROOT/[^/]+/([^/]+)/psf/.*#\1#")
    [ -e "$HSCROOT/$inst/$tract/psf/stars.psf" ] || continue
    [ -e "$HSCROOT/$inst/$tract/psf/psf_residuals_piff.png" ] && continue
    echo "$inst $tract $tract flat $HSCROOT" >> $LOG/work.txt
  done
done
for pref in hst_acs_f814w jwst_nircam_ unwise_ ps1_ lsdr10_ sdss_; do
  for psf in $W/${pref}*/*/psf/stars_*.psf; do
    [ -e "$psf" ] || continue
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
while read -r inst tile suffix flat root; do
  i=$((i+1))
  extra=()
  [ "${flat:-}" = "flat" ] && extra=(--flat --root "$root")
  wroot="${root:-$W}"
  run_to 900 taskpolicy -c utility $PY 73_make_piff_panels.py \
      --instrument "$inst" --tile "$tile" --suffix "$suffix" "${extra[@]}" \
      >>$LOG/last.txt 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -e "$wroot/$inst/$tile/psf/psf_residuals_piff.png" ]; then
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
