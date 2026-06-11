#!/usr/bin/env python
"""
64_step3a_lsdr10_global_saturation.py — GLOBAL per-band saturation onset for
LS DR10 (DECam south), pooled across bricks and anchored to Gaia DR3.

Why global: saturation is a property of the (camera, band) — DECam full well +
~uniform survey exposures — NOT of the brick (a single brick holds only 1-6
bright stars per 0.5-mag bin; pooling ~12 bricks gives hundreds).

Why the Gaia G axis: a saturating star loses core flux, which drags BOTH its
MAG_AUTO fainter AND its peak lower — the point slides ALONG the peak-vs-mag
sequence, hiding the break from any MAG_AUTO-based test (this defeated three
estimator designs).  Gaia G is an external truth axis that saturation cannot
move, so the turnover of median log10(peak) vs G is crisp.

Per band:
  1. pool Gaia-matched point sources (SNR_WIN>8) across the bricks that have
     a 54_ pass-1 catalog (runs SExtractor on extra bricks if < --min-bricks),
  2. G_onset = faint edge of the FAINTEST 0.5-mag step where the growth of
     median log10(peak) drops below 0.65x the photometric reference step
     (median step over G>=16.5), with the median of all brighter steps below
     0.7x ref (persistence against noisy bins),
  3. color   = median(MAG_AUTO - G) of unsaturated stars (G_onset+1 .. +3),
  4. slide   = p95 of (MAG_AUTO - G - color) over SATURATED stars (G<G_onset):
               saturated stars slide faintward in MAG_AUTO, so the catalog
               veto must extend by this margin to catch them,
  5. VETO onset (catalog frame) = G_onset + color + slide.

Validated against the user's per-plot eyeball readings (2026-06-10):
  g 17.12 (user: ~17), r 16.75, i 17.65, z 15.94.

Output: programs_star/csv_saturation/lsdr10_global_onsets.json
        (read automatically by 54_step3a_build_psf_model_lsdr10.py;
        method recorded as 'global_pooled').
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from psf_saturation import core_peak

PROJECT = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS = PROJECT / 'configs'
LS_DIR  = Path('/Volumes/exdisk1/data/DESI_Legacy/COSMOS/dr10/south/coadd')
WORK    = Path('/Volumes/exdisk1/data/photometry_v04')
OUT_JSON = PROJECT / 'programs_star' / 'csv_saturation' / 'lsdr10_global_onsets.json'
GAIA_CSV = Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS/gaia_dr3_cosmos_wide.csv')


def all_bricks():
    return sorted(p.name for p in LS_DIR.glob('*/*') if p.is_dir())


def ensure_pass1(brick, band):
    """Return path to a readable pass-1 catalog, running SExtractor if needed."""
    psf = WORK / f'lsdr10_{band}' / brick / 'psf'
    cat = psf / f'pass1_{brick}.fits'
    if cat.exists():
        try:
            with fits.open(cat) as h:
                _ = h[2].data
            return cat
        except Exception:
            cat.unlink()                      # half-written from a killed run
    psf.mkdir(parents=True, exist_ok=True)
    img_fz = LS_DIR / brick[:3] / brick / f'legacysurvey-{brick}-image-{band}.fits.fz'
    if not img_fz.exists():
        return None
    img_clean = psf / f'image_{brick}.fits'
    if not img_clean.exists():
        with fits.open(img_fz) as h:
            fits.PrimaryHDU(h[1].data.astype(np.float32),
                            header=h[1].header).writeto(img_clean, overwrite=True)
    r = subprocess.run(['sex', str(img_clean),
                        '-c', str(CONFIGS / f'lsdr10_{band}.sex'),
                        '-CATALOG_NAME', str(cat),
                        '-PARAMETERS_NAME', str(CONFIGS / 'pass1_hst_acs.param'),
                        '-FILTER_NAME', str(CONFIGS / 'default.conv'),
                        '-STARNNW_NAME', str(CONFIGS / 'default.nnw'),
                        '-WEIGHT_TYPE', 'NONE', '-MAG_ZEROPOINT', '22.5',
                        '-VERBOSE_TYPE', 'QUIET'], capture_output=True, text=True)
    return cat if (r.returncode == 0 and cat.exists()) else None


def pooled_onset(band, min_bricks, verbose=True):
    import pandas as pd
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    gcat = pd.read_csv(GAIA_CSV)
    cg = SkyCoord(gcat['ra'].values * u.deg, gcat['dec'].values * u.deg)
    G = gcat['phot_g_mean_mag'].values

    bricks = all_bricks()
    have = [b for b in bricks
            if (WORK / f'lsdr10_{band}' / b / 'psf' / f'pass1_{b}.fits').exists()]
    want = [b for b in bricks[::max(1, len(bricks) // max(min_bricks, 1))]
            if b not in have]
    todo = have + want[:max(0, min_bricks - len(have))]
    GG, PP, MM = [], [], []
    frames = []
    used = 0
    for b in todo:
        cat = ensure_pass1(b, band)
        if cat is None:
            continue
        try:
            d = fits.open(cat)[2].data
            img = LS_DIR / b[:3] / b / f'legacysurvey-{b}-image-{band}.fits.fz'
            sci = fits.open(img)[1].data.astype(np.float32)
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
        frames.append({'mag': mau[ok], 'logpk': np.log10(pk[ok])})
        used += 1
    GG = np.concatenate(GG); PP = np.concatenate(PP); MM = np.concatenate(MM)

    ctr, med = [], []
    for lo in np.arange(13.0, 20.5, 0.5):
        m = (GG >= lo) & (GG < lo + 0.5)
        if m.sum() >= 5:
            ctr.append(lo + 0.25); med.append(float(np.median(PP[m])))
    ctr = np.asarray(ctr); med = np.asarray(med)
    if len(med) < 6:
        return None, used, len(GG)
    steps = med[:-1] - med[1:]                      # growth toward brighter
    ref = float(np.median(steps[ctr[:-1] >= 16.5]))
    onset_G = None
    for i in range(len(steps) - 1, -1, -1):         # faint -> bright
        if ctr[i] > 17.5:
            continue
        if steps[i] < 0.65 * ref and float(np.median(steps[:i + 1])) < 0.7 * ref:
            onset_G = float(ctr[i] + 0.5)           # faint edge of the pair
            break                                    # FAINTEST such step
    if onset_G is None:
        return None, used, len(GG)
    un = (GG > onset_G + 1.0) & (GG < onset_G + 3.0) & np.isfinite(MM)
    color = float(np.median(MM[un] - GG[un]))
    sat = (GG < onset_G) & np.isfinite(MM)
    slide = (float(np.percentile(MM[sat] - GG[sat] - color, 95))
             if sat.sum() >= 5 else 0.3)
    veto = onset_G + color + max(slide, 0.0)
    # ceiling constant K (nMgy) for the PER-BRICK onset: median over pooled
    # bricks of each brick's own robust peak-mag line evaluated at the global
    # onset.  (LS images are CALIBRATED nMgy with no per-brick scale keyword,
    # so K is one number per band; per-brick onset variation then enters
    # through each brick's own line = its seeing.)
    Ks = []
    for fr in frames:
        m_, p_ = fr['mag'], fr['logpk']
        fitm = (m_ > veto + 0.5) & (m_ < veto + 4.0) & np.isfinite(m_) & np.isfinite(p_)
        if fitm.sum() < 12:
            continue
        bline = np.polyfit(m_[fitm], p_[fitm], 1)
        if -0.65 <= bline[0] <= -0.25:
            Ks.append(10.0 ** np.polyval(bline, veto))
    K_nmgy = float(np.median(Ks)) if len(Ks) >= 8 else None
    if verbose:
        print(f'  {band}: {used} bricks, {len(GG)} Gaia stars; ref step {ref:+.3f}; '
              f'G_onset={onset_G:.2f} color={color:+.2f} slide_p95={slide:.2f} '
              f'-> VETO onset {veto:.2f}')
    return ({'sat_onset_mag': round(veto, 2), 'gaia_G_onset': onset_G,
             'color_band_minus_G': round(color, 3), 'slide_p95': round(slide, 3),
             'ceiling_K_nmgy': (None if K_nmgy is None else round(K_nmgy, 2)),
             'n_frames_K': len(Ks)},
            used, len(GG))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', default='g,r,i,z')
    ap.add_argument('--min-bricks', type=int, default=12)
    a = ap.parse_args()
    res = {}
    for band in a.bands.split(','):
        print(f'── band {band} ──', flush=True)
        r, nb, ns = pooled_onset(band, a.min_bricks)
        res[band] = dict((r or {'sat_onset_mag': None}),
                         n_bricks_pooled=nb, n_gaia_pooled=ns)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'method': 'pooled Gaia-anchored peak turnover (G axis, immune to the '
                  'MAG_AUTO slide of saturated stars) + band-G color + p95 '
                  'saturated-star slide margin',
        'bands': res}, indent=2))
    print(f'\n[save] {OUT_JSON}')


if __name__ == '__main__':
    main()
