#!/usr/bin/env bash
# 57_run_jwst_psf_qa.sh — reproducible one-command driver for the JWST
# Step-3a PSF QA pipeline on a given tile.
#
# For each of the 4 NIRCam bands it runs:
#   54_step3a_build_psf_model.py   (SExtractor pass-1 → star select → PSFEx)
#   56_make_psf_qa_plots.py        (6 QA diagnostic plots)
# then rebuilds the static QA website with 55_make_psf_qa_site.py.
#
# Per-band parameters are baked into configs/jwst_nircam_<band>.sex
# (notably DEBLEND_MINCONT: 0.02 SW / 0.05 LW) and computed dynamically
# in 54_ (SAMPLE_FWHMRANGE = [0.6,2.5]×PSF_FWHM).  See programs_star/CLAUDE.md
# "Step 3a" and programs_star/v04/HANDOFF_psf_qa.md.
#
# Usage:
#   ./57_run_jwst_psf_qa.sh [TILE]          # default TILE=A4
#   ./57_run_jwst_psf_qa.sh A4 --reuse-pass1   # skip SExtractor pass-1
#
# Full pass-1 ~100 s/band; with --reuse-pass1 only PSFEx+plots (~1 min/band).
set -euo pipefail

TILE="${1:-A4}"
REUSE="${2:-}"          # pass "--reuse-pass1" to skip SExtractor pass-1
PY=/opt/miniconda3/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
BANDS=(f115w f150w f277w f444w)

echo "=== JWST PSF QA — tile ${TILE} ${REUSE} ==="
for b in "${BANDS[@]}"; do
  echo "──── ${b} ────"
  "$PY" "${HERE}/54_step3a_build_psf_model.py" \
      --instrument "jwst_nircam_${b}" --tile "${TILE}" --filter "${b}" ${REUSE}
  "$PY" "${HERE}/56_make_psf_qa_plots.py" \
      --instrument "jwst_nircam_${b}" --tile "${TILE}" --filter "${b}"
done

echo "──── rebuild QA site ────"
"$PY" "${HERE}/55_make_psf_qa_site.py" --clean

echo "=== done — open html/psf_qa/index.html ==="
