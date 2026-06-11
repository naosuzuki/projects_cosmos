#!/usr/bin/env python
"""
54_step3a_build_psf_model_ps1.py — empirical PSF model for Pan-STARRS1 DR1
rings.v3 skycell stacks, porting the HSC/LS-validated ground recipe (wide-Gaia
tilted stellar locus, measured purity limit, 2.5xFWHM isolation, two-pass
PSFEx, global per-band saturation hook).

PS1 specifics (vs LS DR10 bricks):
  * Skycells 6240x6246 @ 0.25"/px (~0.43 deg), fpack CompImageHDU[1]:
        rings.v3.skycell.{proj}.{sub}.stk.{b}.unconv.fits       science
        rings.v3.skycell.{proj}.{sub}.stk.{b}.unconv.mask.fits  uint16 bitmask
    (wt = inverse variance exists; WEIGHT NONE per the validated recipe.)
  * Pixels are ASINH-scaled ("lupton" compression):
        linear = BOFFSET + BSOFTEN * 2*sinh(0.4*ln10 * v)
    -> un-scaled to linear counts before SExtractor.
  * ZP_AB = 25 + 2.5 log10(EXPTIME)  (PS1 stack convention; per-cell EXPTIME).
  * Mask bits (named in the header): SAT=32 -> stellar-core reject;
    BLANK=8 -> gap (off-cell area).
  * Seeing ~1.0-1.3" -> PSF FWHM ~4-5 px: same sampling regime as HSC/LS,
    so configs/psfex_hsc.psfex is reused (SAMPLE_FWHMRANGE dynamic).
  * Saturation onset hook: programs_star/csv_saturation/ps1_global_onsets.json
    (pooled Gaia-anchored, to be measured before mass production); local
    turnover as fallback — PS1 saturates VERY bright (~12-14), so a single
    cell often has no measurable onset and relies on the SAT-bit core veto.

Outputs to /Volumes/exdisk1/data/photometry_v04/ps1_<band>/<proj>_<sub>/psf/.
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
PS1_DIR = Path('/Volumes/exdisk1/data/PanStarrs/COSMOS/skycells/rings.v3.skycell')
WORK    = Path('/Volumes/exdisk1/data/photometry_v04')
PIXSCALE = 0.25
B_SAT, B_BLANK = 32, 8                 # named mask bits (header MSKNAM/MSKVAL)
VIG_HALF = 50                          # pass1_hst_acs.param VIGNET(101,101)

# wide-bbox Gaia superset (no VIS-polygon trim) — same anchor as HSC/LS.
_GAIA_WIDE = Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS/gaia_dr3_cosmos_wide.csv')
GAIA_CSV = (_GAIA_WIDE if _GAIA_WIDE.exists()
            else Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS/gaia_dr3_cosmos.csv'))


def gaia_star_mask(xx, yy, hdr, radius_arcsec=0.6):
    """Boolean mask: detections matched to a Gaia DR3 source within `radius`.
    All-False on any failure → caller falls back to CLASS_STAR."""
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
    p.add_argument('--cell', default='1360_059',
                   help='skycell as <projcell>_<subcell> (default: COSMOS centre)')
    p.add_argument('--filter', default='i', choices=['g', 'r', 'i', 'z', 'y'])
    p.add_argument('--core-box', type=int, default=5)
    p.add_argument('--snr-min', type=float, default=8.0)
    p.add_argument('--nbr-fwhm', type=float, default=2.5,
                   help='Isolation radius in PSF-FWHM units (any detected '
                        'neighbour inside → reject).')
    p.add_argument('--psf-fwhm', type=float, default=2.5,
                   help='PSF model stamp radius in FWHM units (PSF_SIZE).')
    p.add_argument('--reuse-pass1', action='store_true')
    return p.parse_args()


def cell_file(proj, sub, band, ftype=''):
    suffix = f'.{ftype}' if ftype else ''
    f = (PS1_DIR / proj / sub /
         f'rings.v3.skycell.{proj}.{sub}.stk.{band}.unconv{suffix}.fits')
    if not f.exists():
        sys.exit(f'No PS1 file: {f}')
    return f


def main():
    a = parse_args()
    band = a.filter
    proj, sub = a.cell.split('_')
    suffix = a.cell            # output-file key (filter is in the instrument dir)
    inst = f'ps1_{band}'
    out = WORK / inst / a.cell / 'psf'
    out.mkdir(parents=True, exist_ok=True)

    img_fz = cell_file(proj, sub, band)
    msk_fz = cell_file(proj, sub, band, 'mask')
    # un-asinh to LINEAR counts; single clean HDU for SExtractor (its CFITSIO
    # handling of fpack + the asinh scaling is not trustworthy).
    with fits.open(img_fz) as h:
        raw   = h[1].data.astype(np.float64)
        imhdr = h[1].header
    bsoften = float(imhdr.get('BSOFTEN', np.nan))
    boffset = float(imhdr.get('BOFFSET', np.nan))
    if np.isfinite(bsoften) and np.isfinite(boffset):
        sci = (boffset + bsoften * 2.0 * np.sinh(0.4 * np.log(10.0) * raw)
               ).astype(np.float32)
    else:
        sci = raw.astype(np.float32)
    sci[~np.isfinite(sci)] = 0.0
    with fits.open(msk_fz) as h:
        mask = h[1].data.astype(np.int32)
    exptime = float(imhdr.get('EXPTIME', 1.0))
    zp = 25.0 + 2.5 * np.log10(max(exptime, 1e-3))
    img_clean = out / f'image_{suffix}.fits'
    fits.PrimaryHDU(sci, header=imhdr).writeto(img_clean, overwrite=True)
    sex_img = str(img_clean)
    core_reject = B_SAT
    gap_reject  = B_BLANK

    print(f'Instrument : {inst}  (PS1 DR1 rings.v3 stack)')
    print(f'Cell/band  : {proj}.{sub} / {band}   ZP_AB={zp:.3f} '
          f'(=25+2.5log10 EXPTIME={exptime:.0f}s)')
    print(f'Source     : {img_fz.name}')
    print(f'             asinh→linear (BSOFTEN={bsoften:.4g}); WEIGHT NONE; '
          f'mask SAT-bit saturation; SNR>{a.snr_min}')

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
    xi = np.clip(np.round(xx).astype(int), half, nx-half-1)
    yi = np.clip(np.round(yy).astype(int), half, ny-half-1)
    satcore = np.zeros(len(obj), int)
    for k in range(len(obj)):
        box = mask[yi[k]-half:yi[k]+half+1, xi[k]-half:xi[k]+half+1]
        satcore[k] = int(np.sum((box & core_reject) != 0))
    is_pt = (cs > 0.8) & (snr > a.snr_min)
    # GLOBAL per-band onset (pooled, Gaia-anchored — ps1_global_onsets.json,
    # measured by the 64_-style pooled script before mass production); local
    # turnover fallback for the pilot.  Per-source SAT-bit veto always on.
    sat_peak_level, peak_saturated = np.inf, np.zeros(len(obj), bool)
    onset_mag, sat_onset_method = None, None
    gj = PROJECT / 'programs_star' / 'csv_saturation' / 'ps1_global_onsets.json'
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
    saturated = (satcore > 0) | peak_saturated
    if onset_mag is not None and np.isfinite(onset_mag):
        saturated = saturated | (np.isfinite(mag) & (mag < onset_mag))
    print(f'  saturation : {int(saturated.sum())} excluded  '
          f'(onset = {onset_mag if onset_mag is None else round(onset_mag,2)} '
          f'[{sat_onset_method}]; SAT-bit cores {int((satcore>0).sum())})')

    # ── Gaia-DR3-anchored stellar locus ──
    gaia_star = gaia_star_mask(xx, yy, imhdr, radius_arcsec=0.6)
    locus_base = gaia_star & (flg < 2) & (fr > 0) & (snr > a.snr_min) & (~saturated)

    if int(locus_base.sum()) < 10:
        (out / f'psf_{suffix}.meta.json').write_text(json.dumps({
            'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'instrument': inst, 'tile': a.cell, 'filter': band,
            'channel': 'ps1', 'status': 'no_coverage',
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
    mad = max(mad, 0.04)        # small-sample noise floor
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

    # gaps (VIGNET over BLANK) + neighbour contamination
    edge = (flg & 8) > 0
    cheap = (cs > 0.8) & (snr > a.snr_min) & (elon < 1.5) & (fr > 0)
    xiv = np.clip(np.round(xx).astype(int), VIG_HALF, nx-VIG_HALF-1)
    yiv = np.clip(np.round(yy).astype(int), VIG_HALF, ny-VIG_HALF-1)
    gap_masked = np.zeros(len(obj), bool)
    for k in np.where(cheap)[0]:
        box = mask[yiv[k]-VIG_HALF:yiv[k]+VIG_HALF+1, xiv[k]-VIG_HALF:xiv[k]+VIG_HALF+1]
        gap_masked[k] = int(np.sum((box & gap_reject) != 0)) > 5
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
          f'(saturated/bad excl {saturated.sum()}, gap {gap_masked.sum()}, '
          f'isolation (any nbr <{a.nbr_fwhm:.1f}xFWHM={R_nbr*PIXSCALE:.2f}\") '
          f'{int(contaminated.sum())}; PSF_SIZE={psf_size}px)')

    # ── two-pass PSFEx ──
    sel_idx = np.where(star)[0]
    all_cat = out / f'stars_all_{suffix}.fits'
    hcat[2].data = obj[sel_idx]; hcat.writeto(all_cat, overwrite=True)
    fwhm_lo = max(1.2, 0.6 * psf_fwhm_est); fwhm_hi = 2.5 * psf_fwhm_est
    print(f'\n── PSFEx pass 1 (validate; SAMPLE_FWHMRANGE {fwhm_lo:.2f},{fwhm_hi:.2f}) ──', flush=True)
    p1out = out / f'p1outcat_{suffix}.fits'
    r = subprocess.run(['psfex', str(all_cat), '-c', str(CONFIGS / 'psfex_hsc.psfex'),
                        '-PSFVAR_DEGREES', '3',
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
           '-PSFVAR_DEGREES', '3',
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
        'instrument': inst, 'tile': a.cell, 'filter': band, 'channel': 'ps1',
        'zp_ab': float(zp), 'zp_provisional': False, 'pixscale_arcsec': PIXSCALE,
        'exptime_s': exptime,
        'sci_source_file': str(img_fz), 'sci_source_ext': 1,
        'sci_asinh_unscaled': bool(np.isfinite(bsoften)),
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
        'recipe': 'PS1 rings.v3 stack asinh->linear; WEIGHT NONE; mask SAT-bit '
                  'saturation + global/turnover onset; wide-Gaia TILTED locus; '
                  'purity limit; two-pass PSF',
    }
    (out / f'psf_{suffix}.meta.json').write_text(json.dumps(meta, indent=2))
    print(f'\n[save] {out / f"psf_{suffix}.meta.json"}')
    print('=== PS1 PSF model build complete ===')


if __name__ == '__main__':
    main()
