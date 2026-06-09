#!/usr/bin/env bash
# _resume_euclid_x30.sh — finish the interrupted Euclid count-histogram x=30 fix.
#
# At shutdown: A6/f277w fixed; HST 20/20 + JWST 80/80 count histograms are x-max=30;
# Euclid ~19/60 done, ~41 still showing the old auto-scaled axis.  The QA plotter
# needs the pass1 detection catalog (deleted per-tile to save disk), so the only
# way to regenerate those plots is to rebuild -- hence --force.  Re-running all 60
# is simplest and safe (the ~19 already fixed are reproduced identically; models
# are unchanged, only the plots get x=30).
#
#   1) remount /Volumes/exdisk1
#   2) cd ~/github/projects_cosmos
#   3) bash programs_star/v04/_resume_euclid_x30.sh
set -uo pipefail
PY=/opt/miniconda3/bin/python
H=/Users/suzuki/github/projects_cosmos/programs_star/v04
echo "[$(date)] Euclid x=30 count-histogram replot (60 tiles, --force) ..."
"$PY" "$H/60_run_mass_production.py" --missions euclid --force --no-site
echo "[$(date)] rebuilding QA site ..."
"$PY" "$H/55_make_psf_qa_site.py" --clean
echo "[$(date)] DONE — HST/JWST/Euclid count histograms all x-max=30."
