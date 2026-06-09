#!/usr/bin/env bash
PY=/opt/miniconda3/bin/python
H=/Users/suzuki/github/projects_cosmos/programs_star/v04
while pgrep -f '_resume_orchestrate.sh|60_run_mass_production' >/dev/null 2>&1; do
  "$PY" "$H/55_make_psf_qa_site.py" --clean >/dev/null 2>&1; sleep 300; done
"$PY" "$H/55_make_psf_qa_site.py" --clean >/dev/null 2>&1
