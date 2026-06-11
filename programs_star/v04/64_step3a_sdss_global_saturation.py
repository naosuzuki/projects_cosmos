#!/usr/bin/env python
"""
64_step3a_sdss_global_saturation.py — GLOBAL per-band saturation onset for
SDSS DR17 corrected frames, pooled across frames and anchored to Gaia DR3.

Same method as 64_step3a_lsdr10_global_saturation.py (validated against the
user's eyeball reading on LS g): saturation is a (camera, band) property;
the turnover of median log10(core peak) vs **Gaia G** (an external truth
axis that the MAG_AUTO slide of saturating stars cannot move) gives G_onset;
+ median(band-G) colour of unsaturated stars; + p95 MAG_AUTO slide margin of
saturated stars = the catalog-frame VETO onset.

PS1-specific: ensure_pass1 un-asinh-scales the stack to linear counts and
uses the per-cell ZP = 25 + 2.5 log10(EXPTIME) (the 54_ps1 preprocessing);
pre-pass SExtractor runs are parallelised (4 workers) because skycells are
6240^2.  The pass1 catalogs land in the standard psf/ dirs and are REUSED by
the mass production via --reuse-pass1.

Output: programs_star/csv_saturation/ps1_global_onsets.json
        (read automatically by 54_step3a_build_psf_model_ps1.py).
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from psf_saturation import core_peak

PROJECT = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS = PROJECT / 'configs'
SDSS_DIR = Path('/Volumes/exdisk1/data/SDSS/COSMOS/frames')
WORK    = Path('/Volumes/exdisk1/data/photometry_v04')
OUT_JSON = PROJECT / 'programs_star' / 'csv_saturation' / 'sdss_global_onsets.json'
GAIA_CSV = Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS/gaia_dr3_cosmos_wide.csv')


def all_cells():
    out = set()
    for p in SDSS_DIR.glob('*/*/*/frame-i-*.fits.bz2'):
        if p.name.startswith('._'):
            continue
        out.add(p.name.replace('frame-i-', '').replace('.fits.bz2', ''))
    return sorted(out)


def cell_paths(cell, band):
    run6, camcol, field4 = cell.split('-')
    return (SDSS_DIR / '301' / str(int(run6)) / camcol /
            f'frame-{band}-{run6}-{camcol}-{field4}.fits.bz2')


def ensure_pass1(cell, band):
    """Readable pass-1 catalog for (cell, band); runs the 54_ps1 preprocessing
    + SExtractor if needed.  Returns (cat_path, exptime) or (None, None)."""
    psf = WORK / f'sdss_{band}' / cell / 'psf'
    cat = psf / f'pass1_{cell}.fits'
    img_fz = cell_paths(cell, band)
    if not img_fz.exists():
        return None, None
    if cat.exists():
        try:
            with fits.open(cat) as h:
                _ = h[2].data
            with fits.open(img_fz) as h:
                return cat, float(h[1].header.get('EXPTIME', 1.0))
        except Exception:
            cat.unlink()
    psf.mkdir(parents=True, exist_ok=True)
    with fits.open(img_fz) as h:
        sci = h[0].data.astype(np.float32)
        hd = h[0].header
    sci[~np.isfinite(sci)] = 0.0
    exptime, zp = 1.0, 22.5                      # nanomaggies (exact ZP)
    img_clean = psf / f'image_{cell}.fits'
    fits.PrimaryHDU(sci, header=hd).writeto(img_clean, overwrite=True)
    r = subprocess.run(['sex', str(img_clean),
                        '-c', str(CONFIGS / f'sdss_{band}.sex'),
                        '-CATALOG_NAME', str(cat),
                        '-PARAMETERS_NAME', str(CONFIGS / 'pass1_hst_acs.param'),
                        '-FILTER_NAME', str(CONFIGS / 'default.conv'),
                        '-STARNNW_NAME', str(CONFIGS / 'default.nnw'),
                        '-WEIGHT_TYPE', 'NONE', '-MAG_ZEROPOINT', f'{zp:.4f}',
                        '-VERBOSE_TYPE', 'QUIET'], capture_output=True, text=True)
    return (cat, exptime) if (r.returncode == 0 and cat.exists()) else (None, None)


def pooled_onset(band, min_cells, jobs=4, verbose=True):
    import pandas as pd
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    gcat = pd.read_csv(GAIA_CSV)
    cg = SkyCoord(gcat['ra'].values * u.deg, gcat['dec'].values * u.deg)
    G = gcat['phot_g_mean_mag'].values

    cells = all_cells()
    have = [c for c in cells
            if (WORK / f'sdss_{band}' / c / 'psf' / f'pass1_{c}.fits').exists()]
    want = [c for c in cells[::max(1, len(cells) // max(min_cells, 1))]
            if c not in have]
    todo = (have + want)[:max(min_cells, len(have))]
    # parallel pre-pass (SExtractor is single-threaded; cells are independent)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        pre = list(ex.map(lambda c: (c, *ensure_pass1(c, band)), todo))
    GG, PP, MM = [], [], []
    frames = []
    used = 0
    for c, cat, _ in pre:
        if cat is None:
            continue
        try:
            d = fits.open(cat)[2].data
            img = cell_paths(c, band)
            with fits.open(img) as h:
                sci = h[0].data.astype(np.float32)
                nmgy = float(h[0].header.get('NMGY', np.nan))
            sci[~np.isfinite(sci)] = 0.0
        except Exception:
            continue
        snr = np.asarray(d['SNR_WIN'], float)
        mau = np.asarray(d['MAG_AUTO'], float)
        xx = np.asarray(d['X_IMAGE'], float); yy = np.asarray(d['Y_IMAGE'], float)
        cd = SkyCoord(np.asarray(d['ALPHA_J2000'], float) * u.deg,
                      np.asarray(d['DELTA_J2000'], float) * u.deg)
        idx, d2d, _ = cd.match_to_catalog_sky(cg)
        sel = (np.asarray(d2d.arcsec) < 0.6) & (snr > 8)
        pk = core_peak(sci, xx, yy, half=2, idx=np.where(sel)[0])
        ok = sel & (pk > 0)
        GG.append(G[idx[ok]]); PP.append(np.log10(pk[ok])); MM.append(mau[ok])
        frames.append({'cell': c, 'nmgy': nmgy,
                       'mag': mau[ok], 'logpk': np.log10(pk[ok])})
        used += 1
    GG = np.concatenate(GG); PP = np.concatenate(PP); MM = np.concatenate(MM)

    ctr, med = [], []
    for lo in np.arange(11.0, 20.5, 0.5):
        m = (GG >= lo) & (GG < lo + 0.5)
        if m.sum() >= 5:
            ctr.append(lo + 0.25); med.append(float(np.median(PP[m])))
    ctr = np.asarray(ctr); med = np.asarray(med)
    if len(med) < 6:
        return None, used, len(GG)
    steps = med[:-1] - med[1:]
    ref = float(np.median(steps[ctr[:-1] >= 16.0]))
    onset_G = None
    for i in range(len(steps) - 1, -1, -1):
        if ctr[i] > 17.0:
            continue
        if steps[i] < 0.65 * ref and float(np.median(steps[:i + 1])) < 0.7 * ref:
            onset_G = float(ctr[i] + 0.5)
            break
    if onset_G is None:
        return None, used, len(GG)
    un = (GG > onset_G + 1.0) & (GG < onset_G + 3.0) & np.isfinite(MM)
    color = float(np.median(MM[un] - GG[un]))
    # NO slide margin for SDSS — selecting "saturated" stars by G<G_onset
    # conflates saturation displacement with INTRINSIC COLOR SPREAD: a red
    # giant at G=13 has g~15.5 (far below g saturation) yet its large g-G
    # reads as a fake 2-mag "slide" (median slide ~1.5-2 in g/r = colors, not
    # physics).  Slid saturated stars are rejected by the FLUX_RADIUS-bloat /
    # shape cuts anyway.  User-validated on the saturation plots: the unslid
    # onsets match the eyeball (~14 g/r/i, ~12 z); the slid 16.3-16.8 do not.
    slide = 0.0
    veto = onset_G + color + slide
    # ── camera full-well K_DN (counts) for the PER-TILE onset ──────────────
    # Saturation in COUNTS is the camera constant; the calibrated ceiling
    # varies per frame as C = K_DN * NMGY (header counts->nMgy scale), and the
    # onset magnitude additionally varies with seeing through each frame's own
    # peak-mag relation.  Calibrate K_DN once per band: for each pooled frame,
    # fit a robust log10(peak)-vs-mag line on unsaturated stars and evaluate
    # it at the GLOBAL onset -> that frame's implied ceiling; K = median(C/NMGY).
    Ks = []
    for fr in frames:
        m_, p_ = fr['mag'], fr['logpk']
        fitm = (m_ > veto + 0.5) & (m_ < veto + 4.0) & np.isfinite(m_) & np.isfinite(p_)
        if fitm.sum() < 12 or not np.isfinite(fr['nmgy']):
            continue
        bline = np.polyfit(m_[fitm], p_[fitm], 1)
        if not (-0.65 <= bline[0] <= -0.25):
            continue
        C_frame = 10.0 ** np.polyval(bline, veto)
        Ks.append(C_frame / fr['nmgy'])
    K_dn = float(np.median(Ks)) if len(Ks) >= 8 else None
    if verbose:
        print(f'  {band}: {used} cells, {len(GG)} Gaia stars; ref step {ref:+.3f}; '
              f'G_onset={onset_G:.2f} color={color:+.2f} slide_p95={slide:.2f} '
              f'-> VETO onset {veto:.2f}  K_DN={K_dn if K_dn is None else round(K_dn):} ({len(Ks)} frames)')
    return ({'sat_onset_mag': round(veto, 2), 'gaia_G_onset': onset_G,
             'color_band_minus_G': round(color, 3), 'slide_p95': round(slide, 3),
             'fullwell_K_dn': (None if K_dn is None else round(K_dn, 1)),
             'n_frames_K': len(Ks)},
            used, len(GG))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', default='u,g,r,i,z')
    ap.add_argument('--min-cells', type=int, default=48)
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    res = {}
    for band in a.bands.split(','):
        print(f'── band {band} ──', flush=True)
        r, nc, ns = pooled_onset(band, a.min_cells, a.jobs)
        res[band] = dict((r or {'sat_onset_mag': None}),
                         n_cells_pooled=nc, n_gaia_pooled=ns)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'method': 'pooled Gaia-anchored peak turnover (G axis) + band-G color '
                  '+ p95 saturated-star MAG_AUTO slide margin (= LS method)',
        'bands': res}, indent=2))
    print(f'\n[save] {OUT_JSON}')


if __name__ == '__main__':
    main()
