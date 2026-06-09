#!/usr/bin/env bash
# _resume_orchestrate.sh — contention-aware resume of the Step-3a JWST mass
# production after the reboot.  Unlike the linear _resume_after_reboot.sh, this
# OVERLAPS the SW builds with the (network-bound) v1.0 SCI download, then runs
# the heavy LW reads on a clean disk once the download is done — so the single
# HDD is never asked to serve 3 big reads + the download at once (the condition
# that build-failed every LW tile before the shutdown).
#
# Assumes ALREADY RUNNING when launched:
#   * the v1.0 SCI download (05_download_jwst_nircam_extensions.py / aria2c)
#   * a solo A6/f277w LW confirmation build (54_ ... --tile A6 --filter f277w)
set -uo pipefail
PY=/opt/miniconda3/bin/python
H=/Users/suzuki/github/projects_cosmos/programs_star/v04
W=/Volumes/exdisk1/data/photometry_v04
EXCL=A2:f115w,A10:f115w,B4:f115w,B6:f115w
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

log "waiting for the solo A6/f277w LW confirmation build to finish ..."
while pgrep -f '54_step3a.*--tile A6 --filter f277w' >/dev/null 2>&1; do sleep 30; done
if [ -f "$W/jwst_nircam_f277w/A6/psf/psf_f277w.meta.json" ]; then
  LW_OK=1; log "A6/f277w LW build SUCCEEDED — copy-elim LW confirmed end-to-end."
else
  LW_OK=0; log "WARNING: A6/f277w LW build did NOT produce a meta — LW phase will be SKIPPED; debug manually."
fi

# 1) SW good tiles (f115w/f150w) 3-wide — overlaps the still-running download
#    (SW reads + light download writes are proven safe; only LW thrashed).
log "SW builds (f115w/f150w, 3-wide) ..."
"$PY" "$H/60_run_mass_production.py" --missions jwst --bands f115w,f150w --exclude "$EXCL" --no-site

# 2) wait for the v1.0 SCI download (aria2c) so LW + the patched f115w run clean.
log "waiting for v1.0 SCI download (aria2c) to finish ..."
while pgrep -x aria2c >/dev/null 2>&1; do sleep 30; done
"$PY" "$H/05_download_jwst_nircam_extensions.py" --verify-only

# 3) LW bands (f277w/f444w) 2-wide on the now-clean disk.
if [ "$LW_OK" = 1 ]; then
  log "LW builds (f277w/f444w, 2-wide) ..."
  "$PY" "$H/60_run_mass_production.py" --missions jwst --bands f277w,f444w --jobs 2 --no-site
else
  log "SKIPPING LW builds (A6 confirmation failed)."
fi

# 4) the 4 patched f115w tiles, weighted on v1.0 sci+wht (clear stale metas).
log "patched f115w tiles (A2/A10/B4/B6) on v1.0 sci+wht ..."
for t in A2 A10 B4 B6; do rm -f "$W/jwst_nircam_f115w/$t/psf/psf_f115w.meta.json"; done
"$PY" "$H/60_run_mass_production.py" --missions jwst --no-site

# 5) neighbour + radial histograms (both populations) + aggregate.
log "neighbour histograms ..."
"$PY" "$H/61_neighbor_histograms.py" --all --jobs 4
"$PY" "$H/62_aggregate_neighbor_histograms.py"

# 6) final QA site + saturation CSV.
log "final QA site + saturation CSV ..."
"$PY" "$H/55_make_psf_qa_site.py" --clean
"$PY" "$H/collect_saturation_levels.py"
log "ORCHESTRATE COMPLETE"
