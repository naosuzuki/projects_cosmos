#!/usr/bin/env python
"""
64_step3a_unwise_global_saturation.py — global saturation constants for unWISE.

unWISE neo7 coadds have no mask plane; saturated W1/W2 cores pin at a
counts ceiling that varies mildly tile-to-tile with coverage depth
(W1 ~0.7-1.1e5, W2 ~2.0-2.4e5 over the 5 COSMOS tiles).  The per-tile
turnover detector in 54_..._unwise.py finds the break on most tiles but
fails where the bright sequence is sparse (e.g. 1497p015 / 1482p015 in
W2, leaving mag~9.5 pinned stars unflagged).  This script measures the
GLOBAL constants the builder's JSON hook consumes:

  sat_onset_mag    = median of the per-tile turnover onsets (AB)
  ceiling_K_counts = 25th percentile of the per-tile turnover ceilings
                     (conservative: low side, so the per-tile line
                     crossing errs toward flagging)

Per band, per tile: load the builder's surviving pass1 LDAC catalog +
working image, recompute core peaks exactly as the builder does
(core_peak, half = core_box//2 = 2; point sources CLASS_STAR>0.8 &
SNR_WIN>8), run detect_saturation_turnover.  Aggregate across the 5
tiles; also run the detector on the POOLED cloud as a cross-check and
fallback for a band with <2 per-tile successes.

Physics cross-check: WISE saturates at ~8.0 Vega (W1) / ~7.0 Vega (W2)
=> 10.7 / 10.3 AB with VEGA2AB = 2.699 / 3.339.

Output: programs_star/csv_saturation/unwise_global_onsets.json
Usage : python 64_step3a_unwise_global_saturation.py
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
from astropy.io import fits

from psf_saturation import core_peak, detect_saturation_turnover

PROJECT = Path('/Users/suzuki/github/projects_cosmos')
WORK    = Path('/Volumes/exdisk1/data/photometry_v04')
TILES   = ['1482p015', '1497p015', '1497p030', '1512p015', '1512p030']
BANDS   = ['w1', 'w2']
SNR_MIN, HALF = 8.0, 2          # match 54_ builder defaults (core_box=5)
VEGA_SAT = {'w1': 10.7, 'w2': 10.3}   # ~8.0/7.0 Vega + VEGA2AB, cross-check only


def tile_measurement(band: str, tile: str):
    psf = WORK / f'unwise_{band}' / tile / 'psf'
    cat, img = psf / f'pass1_{tile}.fits', psf / f'image_{tile}.fits'
    if not (cat.exists() and img.exists()):
        return None
    with fits.open(cat) as h:
        d = h[2].data
    with fits.open(img) as h:
        sci = np.asarray(h[0].data, float)
    mag = np.asarray(d['MAG_AUTO'], float)          # already AB (runtime ZP)
    is_pt = (np.asarray(d['CLASS_STAR'], float) > 0.8) \
        & (np.asarray(d['SNR_WIN'], float) > SNR_MIN)
    xx = np.asarray(d['XWIN_IMAGE'], float) - 1.0
    yy = np.asarray(d['YWIN_IMAGE'], float) - 1.0
    peak = core_peak(sci, xx, yy, half=HALF, idx=np.where(is_pt)[0])
    ceil_t, onset_t = detect_saturation_turnover(mag, peak, is_pt)
    return dict(mag=mag, peak=peak, is_pt=is_pt,
                ceiling=(None if not np.isfinite(ceil_t) else float(ceil_t)),
                onset=(None if onset_t is None else float(onset_t)))


def main():
    out = {'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'source': '64_step3a_unwise_global_saturation.py '
                     '(per-tile turnover, 5 COSMOS neo7 tiles)',
           'bands': {}}
    for band in BANDS:
        per_tile, pool_m, pool_p, pool_i = {}, [], [], []
        for tile in TILES:
            r = tile_measurement(band, tile)
            if r is None:
                print(f'  [{band} {tile}] missing pass1/image — skipped')
                continue
            per_tile[tile] = {'sat_onset_mag': r['onset'],
                              'ceiling_counts': r['ceiling']}
            pool_m.append(r['mag']); pool_p.append(r['peak'])
            pool_i.append(r['is_pt'])
            print(f"  [{band} {tile}] onset={r['onset']} "
                  f"ceiling={r['ceiling'] and f'{r['ceiling']:.4g}'}")
        ceil_pool, onset_pool = detect_saturation_turnover(
            np.concatenate(pool_m), np.concatenate(pool_p),
            np.concatenate(pool_i))
        onsets = [v['sat_onset_mag'] for v in per_tile.values()
                  if v['sat_onset_mag'] is not None]
        ceils = [v['ceiling_counts'] for v in per_tile.values()
                 if v['ceiling_counts'] is not None]
        if len(onsets) >= 2:
            onset_g, ceil_g, method = (float(np.median(onsets)),
                                       float(np.percentile(ceils, 25)),
                                       'median_per_tile_turnover')
        else:
            onset_g = None if onset_pool is None else float(onset_pool)
            ceil_g = None if not np.isfinite(ceil_pool) else float(ceil_pool)
            method = 'pooled_turnover'
        out['bands'][band] = {
            'sat_onset_mag': onset_g, 'ceiling_K_counts': ceil_g,
            'method': method, 'n_tiles_measured': len(onsets),
            'pooled_onset_mag': None if onset_pool is None else float(onset_pool),
            'expected_onset_physics_AB': VEGA_SAT[band],
            'per_tile': per_tile}
        print(f"[{band}] GLOBAL onset={onset_g and round(onset_g,2)} AB "
              f"({method}, n={len(onsets)}; pooled={onset_pool and round(onset_pool,2)}; "
              f"physics ~{VEGA_SAT[band]})  ceiling_K={ceil_g and f'{ceil_g:.4g}'}")
    dst = PROJECT / 'programs_star' / 'csv_saturation' / 'unwise_global_onsets.json'
    dst.write_text(json.dumps(out, indent=2))
    print(f'wrote {dst}')


if __name__ == '__main__':
    main()
