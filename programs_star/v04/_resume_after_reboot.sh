#!/usr/bin/env bash
# _resume_after_reboot.sh — ONE command to finish the Step-3a mass production
# after a reboot.  Everything is idempotent / skip-done, so partly-finished
# work is NOT redone; it just continues.
#
#   1) mount /Volumes/exdisk1 first (it does not auto-resume the run)
#   2) cd ~/github/projects_cosmos
#   3) bash programs_star/v04/_resume_after_reboot.sh
set -uo pipefail
PY=/opt/miniconda3/bin/python
H=/Users/suzuki/github/projects_cosmos/programs_star/v04
W=/Volumes/exdisk1/data/photometry_v04
EXCL=A2:f115w,A10:f115w,B4:f115w,B6:f115w

echo "=== RESUME $(date) ==="

# 1. finish the v1.0 SCI download for the 4 patched f115w tiles (aria2 --continue
#    resumes the partials; verifies SCI+WHT+ERR at the end).
"$PY" "$H/05_download_jwst_nircam_extensions.py"

# 2a. finish the SW good tiles (f115w/f150w) 3-wide — validated to work.
"$PY" "$H/60_run_mass_production.py" --missions jwst --bands f115w,f150w --exclude "$EXCL" --no-site
# 2b. LW bands (f277w/f444w) at LOWER concurrency.  They build_failed during
#     the pre-shutdown run while the one HDD was ALSO serving the v1.0 download
#     + 3-wide SW + a diagnostic (I/O contention; ~960 s per LW read).  With the
#     download finished and only 2-wide they should complete.  If any LW tile
#     STILL build_fails, run it solo to capture the real SExtractor/PSFEx error:
#       "$PY" "$H/54_step3a_build_psf_model.py" --instrument jwst_nircam_f277w --tile A1 --filter f277w
"$PY" "$H/60_run_mass_production.py" --missions jwst --bands f277w,f444w --jobs 2 --no-site

# 3. the 4 patched tiles, weighted on the v1.0 sci+wht (clear their stale
#    v0_8/weightless metas so they re-run on v1.0).
for t in A2 A10 B4 B6; do rm -f "$W/jwst_nircam_f115w/$t/psf/psf_f115w.meta.json"; done
"$PY" "$H/60_run_mass_production.py" --missions jwst --no-site

# 4. neighbour-count + radial histograms (all tiles, both populations) + aggregate
"$PY" "$H/61_neighbor_histograms.py" --all --jobs 4
"$PY" "$H/62_aggregate_neighbor_histograms.py"

# 5. final QA site + saturation CSV
"$PY" "$H/55_make_psf_qa_site.py" --clean
"$PY" "$H/collect_saturation_levels.py"
echo "=== RESUME COMPLETE $(date) ==="
