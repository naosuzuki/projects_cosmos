#!/usr/bin/env python
"""
54_step3a_build_psf_model_sdss.py — empirical PSF model for SDSS DR17
corrected frames, porting the HSC/LS/PS1-validated ground recipe (wide-Gaia
tilted stellar locus, measured purity limit, 2.5xFWHM isolation, two-pass
PSFEx, global per-band saturation hook).

SDSS specifics (vs PS1 skycells):
  * Corrected frames 2048x1489 @ 0.396"/px (13.6'x9.8'), bz2-compressed,
    HDU0 = sky-subtracted CALIBRATED image in nanomaggies -> ZP_AB = 22.5
    EXACT, no rescaling needed (same convention as LS DR10).
  * NO mask plane in the frame product (fpM not downloaded): saturated
    pixels are interpolated upstream, so the per-source SAT-bit veto is
    unavailable — the bright limit comes from the GLOBAL pooled Gaia
    turnover onset (sdss_global_onsets.json; 64_-style) with the local
    turnover as pilot fallback, plus the FLUX_RADIUS-bloat/shape cuts.
  * Frames are SMALL (~0.037 deg^2, ~30-100 usable stars): PSFVAR_DEGREES 1
    (a quadratic over a drift-scan frame overfits with so few stars).
  * Bands: ugriz (u is shallow — expect no_coverage stubs on some frames).

Outputs to /Volumes/exdisk1/data/photometry_v04/sdss_<band>/<run-camcol-field>/psf/.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from psf_saturation import core_peak, detect_saturation_turnover

PROJECT = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS = PROJECT / 'configs'
SDSS_DIR = Path('/Volumes/exdisk1/data/SDSS/COSMOS/frames')
WORK    = Path('/Volumes/exdisk1/data/photometry_v04')
PIXSCALE = 0.396
ZP_AB    = 22.5                       # nanomaggy zeropoint (exact)
VIG_HALF = 50

_GAIA_WIDE = Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS/gaia_dr3_cosmos_wide.csv')
GAIA_CSV = (_GAIA_WIDE if _GAIA_WIDE.exists()
            else Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS/gaia_dr3_cosmos.csv'))


def gaia_star_mask(xx, yy, hdr, radius_arcsec=0.6):
    """Boolean mask: detections matched to a Gaia DR3 source within `radius`."""
    try:
        import pandas as pd
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        g = pd.read_csv(GAIA_CSV)
        rc = next(c for c in g.columns if c.lower() in ('ra', 'ra_deg', 'ra_icrs'))
        dc = next(c for c in g.columns if c.lower() in ('dec', 'dec_deg', 'dec_icrs'))
        w = WCS(hdr)
        ra, dec = w.all_pix2world(xx, yy, 1)
        cd = SkyCoord(ra * u.deg, dec * u.deg)
        cg = SkyCoord(g[rc].values * u.deg, g[dc].values * u.deg)
        _, d2d, _ = cd.match_to_catalog_sky(cg)
        return np.asarray(d2d.arcsec < radius_arcsec)
    except Exception as e:
        print(f'  [warn] Gaia match unavailable ({type(e).__name__}: {e}); '
              f'falling back to CLASS_STAR locus')
        return np.zeros(len(xx), bool)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--frame', default='001462-4-0165',
                   help='frame as <run6>-<camcol>-<field4> (default: COSMOS centre)')
    p.add_argument('--filter', default='i', choices=['u', 'g', 'r', 'i', 'z'])
    p.add_argument('--core-box', type=int, default=5)
    p.add_argument('--snr-min', type=float, default=8.0)
    p.add_argument('--nbr-fwhm', type=float, default=2.5)
    p.add_argument('--psf-fwhm', type=float, default=2.5)
    p.add_argument('--reuse-pass1', action='store_true')
    return p.parse_args()


def frame_file(frame, band):
    run6, camcol, field4 = frame.split('-')
    run = str(int(run6))
    for rerun in ('301',):
        f = (SDSS_DIR / rerun / run / camcol /
             f'frame-{band}-{run6}-{camcol}-{field4}.fits.bz2')
        if f.exists():
            return f
    sys.exit(f'No SDSS frame for {frame} band {band}')


def main():
    a = parse_args()
    band, frame = a.filter, a.frame
    suffix = frame
    inst = f'sdss_{band}'
    out = WORK / inst / frame / 'psf'
    out.mkdir(parents=True, exist_ok=True)

    img_bz = frame_file(frame, band)
    with fits.open(img_bz) as h:
        sci   = h[0].data.astype(np.float32)     # nanomaggies, sky-subtracted
        imhdr = h[0].header
    sci[~np.isfinite(sci)] = 0.0
    zp = ZP_AB
    img_clean = out / f'image_{suffix}.fits'
    fits.PrimaryHDU(sci, header=imhdr).writeto(img_clean, overwrite=True)
    sex_img = str(img_clean)

    print(f'Instrument : {inst}  (SDSS DR17 corrected frame)')
    print(f'Frame/band : {frame} / {band}   ZP_AB={zp:.1f} (nanomaggy, exact)')
    print(f'Source     : {img_bz.name}')
    print(f'             HDU0 nanomaggies; WEIGHT NONE; no mask plane '
          f'(global/turnover onset only); SNR>{a.snr_min}')

    # ── 1. SExtractor pass 1 ──
    cat1 = out / f'pass1_{suffix}.fits'
    if not (a.reuse_pass1 and cat1.exists()):
        print('\n── SExtractor pass 1 (detection + VIGNET) ──', flush=True)
        cmd = ['sex', sex_img,
               '-c', str(CONFIGS / f'{inst}.sex'),
               '-CATALOG_NAME', str(cat1),
               '-PARAMETERS_NAME', str(CONFIGS / 'pass1_hst_acs.param'),
               '-FILTER_NAME', str(CONFIGS / 'default.conv'),
               '-STARNNW_NAME', str(CONFIGS / 'default.nnw'),
               '-WEIGHT_TYPE', 'NONE',
               '-MAG_ZEROPOINT', f'{zp:.4f}',
               '-VERBOSE_TYPE', 'QUIET']
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-1500:], file=sys.stderr); sys.exit('SExtractor pass1 failed')
        print(f'  wall {time.time()-t0:.1f}s')

    # ── 2. star selection ──
    print('\n── star selection ──', flush=True)
    hcat = fits.open(cat1); obj = hcat[2].data
    mag  = np.asarray(obj['MAG_AUTO'], float); fr = np.asarray(obj['FLUX_RADIUS'], float)
    fwhm = np.asarray(obj['FWHM_IMAGE'], float); cs = np.asarray(obj['CLASS_STAR'], float)
    snr  = np.asarray(obj['SNR_WIN'], float);   flg = np.asarray(obj['FLAGS'], int)
    elon = np.asarray(obj['ELONGATION'], float)
    ell  = np.asarray(obj['ELLIPTICITY'], float)
    xx   = np.asarray(obj['X_IMAGE'], float);   yy = np.asarray(obj['Y_IMAGE'], float)

    ny, nx = sci.shape
    half = a.core_box // 2
    satcore = np.zeros(len(obj), int)            # no mask plane in SDSS frames
    is_pt = (cs > 0.8) & (snr > a.snr_min)
    sat_peak_level, peak_saturated = np.inf, np.zeros(len(obj), bool)
    onset_mag, sat_onset_method = None, None
    gj = PROJECT / 'programs_star' / 'csv_saturation' / 'sdss_global_onsets.json'
    if gj.exists():
        gv = json.loads(gj.read_text()).get('bands', {}).get(band, {})
        if gv.get('sat_onset_mag') is not None:
            onset_mag = float(gv['sat_onset_mag'])
            sat_onset_method = 'global_pooled'
    if onset_mag is None:
        peak = core_peak(sci, xx, yy, half=half, idx=np.where(is_pt)[0])
        sat_peak_level, onset_mag = detect_saturation_turnover(mag, peak, is_pt)
        if onset_mag is not None:
            sat_onset_method = 'turnover_local'
        peak_saturated = np.isfinite(peak) & (peak >= sat_peak_level)
    saturated = peak_saturated.copy()
    if onset_mag is not None and np.isfinite(onset_mag):
        saturated = saturated | (np.isfinite(mag) & (mag < onset_mag))
    print(f'  saturation : {int(saturated.sum())} excluded  '
          f'(onset = {onset_mag if onset_mag is None else round(onset_mag,2)} '
          f'[{sat_onset_method}]; no mask plane)')

    # ── Gaia-DR3-anchored stellar locus ──
    gaia_star = gaia_star_mask(xx, yy, imhdr, radius_arcsec=0.6)
    locus_base = gaia_star & (flg < 2) & (fr > 0) & (snr > a.snr_min) & (~saturated)

    if int(locus_base.sum()) < 10:
        (out / f'psf_{suffix}.meta.json').write_text(json.dumps({
            'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'instrument': inst, 'tile': frame, 'filter': band,
            'channel': 'sdss', 'status': 'no_coverage',
            'n_locus_stars': int(locus_base.sum()), 'n_model_stars': 0,
        }, indent=2))
        print(f'  [no_coverage] only {int(locus_base.sum())} locus stars — '
              f'wrote stub meta and exiting cleanly')
        return

    anchored = int(locus_base.sum()) >= 8
    if not anchored:
        locus_base = ((cs > 0.8) & (snr > a.snr_min) & (flg < 2) & (elon < 1.5)
                      & (fr > 0) & (~saturated))
    med = float(np.median(fr[locus_base]))
    mad = 1.4826 * float(np.median(np.abs(fr[locus_base] - med)))
    for _ in range(20):
        sel = np.abs(fr[locus_base] - med) < 3.0 * mad
        if sel.sum() < 5: break
        nmed = float(np.median(fr[locus_base][sel]))
        nmad = 1.4826 * float(np.median(np.abs(fr[locus_base][sel] - nmed)))
        if abs(nmed-med) < 1e-4 and abs(nmad-mad) < 1e-4: med, mad = nmed, nmad; break
        med, mad = nmed, nmad
    mad = max(mad, 0.04)
    onloc = locus_base & (np.abs(fr - med) < 3.0 * mad)
    psf_fwhm_est = (float(np.median(fwhm[onloc])) if onloc.sum() >= 3
                    else float(np.median(fwhm[locus_base])))
    fwhm_min = 0.5 * psf_fwhm_est
    std = float(np.std(fr[locus_base]))
    cut = med - 3.0 * mad
    upper = med + 3.0 * mad
    print(f'  locus anchor : {"Gaia DR3" if anchored else "CLASS_STAR (Gaia<8)"}'
          f'  ({int(gaia_star.sum())} Gaia matched, {int(locus_base.sum())} locus stars)')
    print(f'  PSF FWHM est = {psf_fwhm_est:.2f} px ; stellar locus = {med:.3f} px '
          f'(MAD-σ {mad:.3f}) ; artifact cut = {cut:.3f} px')

    bright_pivot = ((onset_mag + 4.0) if (onset_mag is not None and np.isfinite(onset_mag))
                    else (float(np.percentile(mag[locus_base], 25)) if locus_base.sum() else 99.0))

    # neighbour contamination (no gap test — frames are fully covered)
    edge = (flg & 8) > 0
    gap_masked = np.zeros(len(obj), bool)
    tree = cKDTree(np.column_stack([xx, yy]))
    fwhm_px = 2.0 * med
    R_nbr   = a.nbr_fwhm * fwhm_px
    contaminated = np.zeros(len(obj), bool)
    cand = (cs > 0.8) & (snr > a.snr_min) & (fr > 0)
    for k in np.where(cand)[0]:
        for j in tree.query_ball_point([xx[k], yy[k]], R_nbr):
            if j != k:
                contaminated[k] = True; break
    psf_size = int(round(2.0 * a.psf_fwhm * fwhm_px)) | 1
    e_round = ell < 0.20

    upper_bright = med + 6.0 * mad

    # ── TILTED (magnitude-dependent) stellar locus — Gaia-anchored ──
    slope, icpt, mad_t, tilted = 0.0, med, mad, False
    tm_all, tf_all = mag[locus_base], fr[locus_base]
    tok = np.isfinite(tm_all) & np.isfinite(tf_all)
    if anchored and int(tok.sum()) >= 20 and (np.ptp(tm_all[tok]) >= 1.5):
        tm, tf = tm_all[tok], tf_all[tok]
        kp, bb, ss = np.ones(tm.size, bool), (0.0, med), mad
        for _ in range(8):
            if kp.sum() < 8:
                break
            bb = np.polyfit(tm[kp], tf[kp], 1)
            rr = tf - np.polyval(bb, tm)
            ss = 1.4826 * float(np.median(np.abs(rr[kp] - np.median(rr[kp]))))
            if ss <= 0:
                break
            kp = np.abs(rr) < 3.5 * ss
        slope = float(np.clip(bb[0], -0.08, 0.02))
        icpt, mad_t, tilted = float(bb[1]), max(float(ss), 0.030), True
    locus_of = lambda mm: slope * np.asarray(mm, float) + icpt

    # ── MEASURED purity faint limit ──
    wpx = max(0.40, 8.0 * mad)
    pure_to = None
    probe = ((cs > 0.8) & (snr > a.snr_min) & (flg < 2) & (fr > 0)
             & np.isfinite(mag) & (np.abs(fr - med) < wpx) & (~saturated))
    m_lo = (onset_mag if (onset_mag is not None and np.isfinite(onset_mag))
            else float(np.nanpercentile(mag[probe], 1)))
    for lo in np.arange(np.floor(m_lo * 2) / 2, 26.0, 0.5):
        b = probe & (mag >= lo) & (mag < lo + 0.5)
        if b.sum() < 8:
            continue
        if abs(float(np.median(fr[b])) - float(locus_of(lo + 0.25))) > 2.0 * mad:
            pure_to = float(lo)
            break
    pure_arr = (mag < pure_to) if pure_to is not None else np.ones(len(obj), bool)

    cut_arr   = locus_of(mag) - 3.0 * mad_t
    upper_arr = np.where(np.isfinite(mag) & (mag < bright_pivot),
                         locus_of(mag) + 6.0 * mad_t, locus_of(mag) + 3.0 * mad_t)
    keep_fr   = (fr > cut_arr) & (fr < upper_arr)
    print(f'  tilted locus : {"on (Gaia-fit)" if tilted else "OFF(const)"}  '
          f'FR={slope:+.4f}·mag{icpt:+.3f}px  ({slope*PIXSCALE*1000:+.1f} mas/mag)  '
          f'MAD_t={mad_t:.4f}px')
    print(f'  purity limit : '
          + (f'mag < {pure_to:.1f} (measured)' if pure_to is not None
             else 'none (band stays stellar to the SNR floor)'))
    star = ((cs > 0.8) & (snr > a.snr_min) & (elon < 1.5) & (fwhm > fwhm_min)
            & keep_fr & e_round & pure_arr
            & (~saturated) & (~edge) & (~contaminated) & (~gap_masked))
    print(f'  PSF stars selected : {star.sum()}   '
          f'(saturated excl {saturated.sum()}, '
          f'isolation (any nbr <{a.nbr_fwhm:.1f}xFWHM={R_nbr*PIXSCALE:.2f}\") '
          f'{int(contaminated.sum())}; PSF_SIZE={psf_size}px)')

    # ── two-pass PSFEx (PSFVAR_DEGREES 1 — small drift-scan frames) ──
    sel_idx = np.where(star)[0]
    all_cat = out / f'stars_all_{suffix}.fits'
    hcat[2].data = obj[sel_idx]; hcat.writeto(all_cat, overwrite=True)
    fwhm_lo = max(1.2, 0.6 * psf_fwhm_est); fwhm_hi = 2.5 * psf_fwhm_est
    print(f'\n── PSFEx pass 1 (validate; SAMPLE_FWHMRANGE {fwhm_lo:.2f},{fwhm_hi:.2f}) ──', flush=True)
    p1out = out / f'p1outcat_{suffix}.fits'
    r = subprocess.run(['psfex', str(all_cat), '-c', str(CONFIGS / 'psfex_hsc.psfex'),
                        '-PSFVAR_DEGREES', '1',
                        '-SAMPLE_FWHMRANGE', f'{fwhm_lo:.2f},{fwhm_hi:.2f}',
                        '-SAMPLE_MINSN', f'{a.snr_min:.1f}',
                        '-PSF_SIZE', f'{psf_size},{psf_size}',
                        '-OUTCAT_TYPE', 'FITS_LDAC', '-OUTCAT_NAME', str(p1out),
                        '-CHECKIMAGE_TYPE', 'NONE'], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], file=sys.stderr); sys.exit('PSFEx pass1 failed')
    oc1 = fits.open(p1out)
    od1 = next(h.data for h in oc1 if h.data is not None
               and getattr(h, 'columns', None) is not None and len(h.columns) > 2)
    fpsf = np.asarray(od1['FLAGS_PSF'], int)
    axy = np.column_stack([np.asarray(od1['X_IMAGE'], float)[fpsf == 0],
                           np.asarray(od1['Y_IMAGE'], float)[fpsf == 0]])
    s_x = xx[sel_idx]; s_y = yy[sel_idx]
    if len(axy):
        dd, _ = cKDTree(axy).query(np.column_stack([s_x, s_y])); s_acc = dd < 1.0
    else:
        s_acc = np.zeros(len(sel_idx), bool)
    s_mag = mag[sel_idx]; s_fr = fr[sel_idx]; s_fwhm = fwhm[sel_idx]; s_ell = ell[sel_idx]
    s_override = ((~s_acc) & np.isfinite(s_mag) & (s_mag < bright_pivot)
                  & (s_fr > cut) & (s_fr < upper_bright)
                  & (np.abs(s_fwhm - psf_fwhm_est) < 0.15 * psf_fwhm_est)
                  & (s_ell < 0.10))
    model_mask = s_acc | s_override
    n_acc = int(s_acc.sum()); n_ovr = int(s_override.sum()); n_mod = int(model_mask.sum())
    print(f'  pass1: PSFEx-accepted {n_acc}, bright shape-override {n_ovr} → model set {n_mod}')

    star_cat = out / f'stars_{suffix}.fits'
    hcat[2].data = obj[sel_idx][model_mask]; hcat.writeto(star_cat, overwrite=True)
    print(f'  wrote {star_cat.name} ({n_mod} model stars: {n_acc} PSFEx + {n_ovr} bright anchors)')

    acc2 = '0.5' if n_ovr > 0 else '0.01'
    print(f'── PSFEx pass 2 (final model; PSF_ACCURACY {acc2}) ──', flush=True)
    cmd = ['psfex', str(star_cat), '-c', str(CONFIGS / 'psfex_hsc.psfex'),
           '-PSFVAR_DEGREES', '1',
           '-SAMPLE_FWHMRANGE', f'{fwhm_lo:.2f},{fwhm_hi:.2f}',
           '-SAMPLE_MINSN', f'{a.snr_min:.1f}', '-PSF_ACCURACY', acc2,
           '-PSF_SIZE', f'{psf_size},{psf_size}',
           '-CHECKIMAGE_TYPE', 'RESIDUALS,PROTOTYPES,SNAPSHOTS,SAMPLES',
           '-CHECKIMAGE_NAME',
           f'{out}/resi.fits,{out}/proto.fits,{out}/snap.fits,{out}/samp.fits']
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], file=sys.stderr); sys.exit('PSFEx pass2 failed')
    psf = star_cat.with_suffix('.psf')
    if not psf.exists():
        alt = Path.cwd() / psf.name
        if alt.exists(): shutil.move(alt, psf)
    ph = fits.open(psf)[1].header
    for f in (all_cat, p1out, all_cat.with_suffix('.psf')):
        try: f.unlink()
        except OSError: pass
    print(f'  wall {time.time()-t0:.1f}s')
    print(f'  PSF model: chi2={ph.get("CHI2",-1):.3f}  FWHM={ph.get("PSF_FWHM",-1):.2f} px  '
          f'accepted={ph.get("ACCEPTED","?")}')

    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'instrument': inst, 'tile': frame, 'filter': band, 'channel': 'sdss',
        'zp_ab': float(zp), 'zp_provisional': False, 'pixscale_arcsec': PIXSCALE,
        'sci_source_file': str(img_bz), 'sci_source_ext': 0,
        'psf_fwhm_est_px': float(psf_fwhm_est),
        'sample_fwhmrange': [float(fwhm_lo), float(fwhm_hi)],
        'stellar_locus_px': float(med), 'locus_std_px': float(std),
        'locus_mad_sigma_px': float(mad), 'artifact_cut_px': float(cut),
        'upper_px': float(upper), 'upper_bright_px': float(upper_bright),
        'bright_pivot_mag': float(bright_pivot),
        'stellar_locus_arcsec': float(med * PIXSCALE),
        'locus_mad_arcsec': float(mad * PIXSCALE),
        'locus_tilted': bool(tilted),
        'locus_slope_px_per_mag': float(slope),
        'locus_intercept_px': float(icpt),
        'locus_mad_tilt_px': float(mad_t),
        'locus_slope_mas_per_mag': float(slope * PIXSCALE * 1000.0),
        'faint_purity_mag': pure_to,
        'locus_anchor': ('gaia_dr3' if anchored else 'class_star'),
        'n_gaia_matched': int(gaia_star.sum()),
        'sat_onset_mag': onset_mag,
        'sat_onset_method': sat_onset_method,
        'sat_peak_level': (None if not np.isfinite(sat_peak_level)
                           else float(sat_peak_level)),
        'n_peak_saturated': int(peak_saturated.sum()),
        'snr_min': float(a.snr_min),
        'n_psf_stars': int(star.sum()),
        'n_psfex_accepted': n_acc,
        'n_override_bright': n_ovr,
        'n_model_stars': n_mod,
        'psf_accuracy_pass2': float(acc2),
        'n_saturated_excluded': int(saturated.sum()),
        'n_gap_masked_excluded': int(gap_masked.sum()),
        'n_contaminated_excluded': int(contaminated.sum()),
        'nbr_fwhm': float(a.nbr_fwhm),
        'psf_fwhm': float(a.psf_fwhm),
        'nbr_radius_arcsec': float(R_nbr * PIXSCALE),
        'fwhm_px': float(fwhm_px),
        'psf_size_px': int(psf_size),
        'psf_chi2': float(ph.get('CHI2', -1)), 'psf_fwhm_px': float(ph.get('PSF_FWHM', -1)),
        'psf_accepted': int(ph.get('ACCEPTED', -1)), 'psf_model': str(psf),
        'recipe': 'SDSS DR17 corrected frame (nanomaggy, sky-subtracted); WEIGHT '
                  'NONE; global/turnover onset (no mask plane); wide-Gaia TILTED '
                  'locus; purity limit; two-pass PSF, PSFVAR_DEGREES 1',
    }
    (out / f'psf_{suffix}.meta.json').write_text(json.dumps(meta, indent=2))
    print(f'\n[save] {out / f"psf_{suffix}.meta.json"}')
    print('=== SDSS PSF model build complete ===')


if __name__ == '__main__':
    main()
