#!/usr/bin/env python
"""collect_saturation_levels.py — scan every psf_<tile>.meta.json produced by
the single-HDU PSF builders (HST ACS, Euclid VIS) and record the detected
saturation level per (instrument, tile) into one CSV.

A tile's saturation magnitude is the `sat_onset_mag` written by 54_*:
  - a finite value  → saturation IS present (HST plateau, or a genuinely
    bright masked-core population)
  - empty / NaN     → no saturation detected for the PSF candidates

Output: /Volumes/exdisk1/data/photometry_v04/saturation_levels.csv
"""
from __future__ import annotations
import csv, json
from pathlib import Path

WORK = Path('/Volumes/exdisk1/data/photometry_v04')   # PSF outputs (scratch disk)
# Saturation records are kept IN THE REPO (git-tracked), not on the scratch disk.
OUTDIR = Path('/Users/suzuki/github/projects_cosmos/programs_star/csv_saturation')
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT  = OUTDIR / 'saturation_levels.csv'
INSTRUMENTS = ['hst_acs_f814w', 'euclid_vis']

rows = []
for inst in INSTRUMENTS:
    root = WORK / inst
    if not root.exists():
        continue
    for meta in sorted(root.glob('*/psf/psf_*.meta.json')):
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        onset = m.get('sat_onset_mag')
        rows.append({
            'instrument':        m.get('instrument', inst),
            'tile':              m.get('tile', meta.parent.parent.name),
            'saturated':         'yes' if onset is not None else 'no',
            'sat_onset_mag':     ('' if onset is None else round(float(onset), 3)),
            'sat_peak_level':    ('' if m.get('sat_peak_level') is None
                                  else round(float(m['sat_peak_level']), 4)),
            'n_peak_saturated':  m.get('n_peak_saturated', ''),
            'n_saturated_excluded': m.get('n_saturated_excluded', ''),
            'zp_ab':             round(float(m.get('zp_ab', 0)), 4),
            'created_utc':       m.get('created_utc_iso', ''),
        })

cols = ['instrument', 'tile', 'saturated', 'sat_onset_mag', 'sat_peak_level',
        'n_peak_saturated', 'n_saturated_excluded', 'zp_ab', 'created_utc']
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

nsat = sum(1 for r in rows if r['saturated'] == 'yes')
print(f'wrote {OUT}  ({len(rows)} tiles; {nsat} with detected saturation)')
for r in rows:
    print(f"  {r['instrument']:14s} {r['tile']:12} "
          f"saturated={r['saturated']:3} onset_mag={r['sat_onset_mag']}")
