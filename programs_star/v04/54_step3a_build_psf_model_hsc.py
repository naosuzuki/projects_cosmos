#!/usr/bin/env python
"""
54_step3a_build_psf_model_hsc.py — empirical PSF model for HSC SSP (s23b)
deepCoadd patches, adapting the validated JWST/Euclid/HST Step-3a recipe to
the LSST-pipeline coadd format.

HSC specifics (vs the COSMOS-Web missions):
  * Multi-extension LSST `deepCoadd_calexp`:  IMAGE=[1] / MASK=[2] / VARIANCE=[3].
    SExtractor reads the IMAGE directly, weighted by the VARIANCE plane
    (WEIGHT_TYPE MAP_VAR) — no redundant copies.
  * Saturation/bad pixels come from the MASK plane bits (NOT a sci==0 trick):
        bit 1 = SAT, bit 0 = BAD  -> reject in the stellar core
        bit 8 = NO_DATA, bit 4 = EDGE -> "gap" in the VIGNET footprint
    (cleaner than the JWST masked-core==0 / HST peak-plateau heuristics).
  * Pixel scale 0.168"/px, seeing ~0.6" (PSF FWHM ~3.6 px).
  * ZP: the modern stack stores calibration in a PhotoCalib object; the images
    are sky-subtracted counts.  We use the hsc_<band>.sex nominal ZP=27.0 for
    the mag axis -- PROVISIONAL; the PSF *shape* is ZP-independent.  (A later
    pass can pin ZP empirically against the HSC catalog i_psf mags.)

Reuses configs/hsc_<band>.sex + pass1_hst_acs.param (VIGNET 101) +
psfex_hst_acs.psfex (PIXEL_AUTO, 2x oversampled) — all ground-resolution-tuned.

Star selection (identical to the locked recipe): CLASS_STAR>0.8, SNR_WIN>100,
ELONGATION<1.5, ELLIPTICITY<0.20, FWHM>0.5*PSF_FWHM, iterative-MAD FLUX_RADIUS
locus, neighbour cut (n_1>=2 AND FLAGS<2), minus MASK-flagged saturated/gap.

Outputs to /Volumes/exdisk1/data/photometry_v04/hsc_<band>/<patch>/psf/.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

PROJECT = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS = PROJECT / 'configs'
HSC_DIR = Path('/Volumes/exdisk1/data/HSC/COSMOS/s23b')
WORK    = Path('/Volumes/exdisk1/data/photometry_v04')
TRACT   = 9813
PIXSCALE = 0.168
ZP_NOMINAL = 27.0                     # hsc_<band>.sex MAG_ZEROPOINT (provisional)
# LSST mask-plane bit assignments (from the calexp header MP_* keywords)
B_BAD, B_SAT, B_EDGE, B_NODATA = 0, 1, 4, 8
CORE_REJECT = (1 << B_SAT) | (1 << B_BAD)            # saturated/bad in stellar core
GAP_REJECT  = (1 << B_NODATA) | (1 << B_EDGE)        # coverage hole/edge in VIGNET
VIG_HALF = 50                                        # pass1_hst_acs.param VIGNET(101,101)


GAIA_CSV = Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS/gaia_dr3_cosmos.csv')


def gaia_star_mask(xx, yy, hdr, radius_arcsec=0.6):
    """Boolean mask: detections matched to a Gaia DR3 source within `radius`.
    COSMOS is galaxy-rich, so CLASS_STAR>0.8 admits compact galaxies that bias
    the FLUX_RADIUS locus high; anchoring to Gaia point sources gives the true
    tight locus.  All-False on any failure → caller falls back to CLASS_STAR."""
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
    p.add_argument('--patch', default='40', help='HSC tract-9813 patch number (0-80)')
    p.add_argument('--filter', default='i', choices=['g', 'r', 'i', 'z', 'y'])
    p.add_argument('--core-box', type=int, default=5)
    p.add_argument('--snr-min', type=float, default=8.0,
                   help='SNR_WIN floor for PSF stars.  HSC WEIGHT-NONE SNR_WIN '
                        'is on a compressed scale (max ~250); 8 reaches the '
                        'faint end of the clean stellar locus (~mag 24).')
    p.add_argument('--nbr-fwhm', type=float, default=2.5,
                   help='Neighbour radius in units of PSF FWHM.  Reject a PSF star '
                        'if ANY detected object lies within this radius (no '
                        'brightness gate — inside the fit core even a faint '
                        'neighbour matters).  2.5xFWHM = the model stamp; distant '
                        'galaxies are outside the (small) model so they are '
                        'harmless (strict wider isolation would gut the COSMOS '
                        'sample: 6" -> 3%% of stars).')
    p.add_argument('--psf-fwhm', type=float, default=2.5,
                   help='PSF model stamp radius in FWHM units (PSF_SIZE).')
    p.add_argument('--nbr-dmag', type=float, default=2.5,
                   help='Brightness gate: only neighbours within --nbr-radius that '
                        'are <this many mag fainter than the star contaminate.  '
                        'Avoids rejecting stars next to faint background galaxies '
                        '(in galaxy-dense COSMOS ~half have a faint object <2").')
    p.add_argument('--reuse-pass1', action='store_true')
    return p.parse_args()


def find_calexp(patch, band):
    cands = [p for p in HSC_DIR.rglob(
                f'deepCoadd_calexp_{TRACT}_{patch}_{band}_*.fits')
             if not p.name.startswith('._')]
    if not cands:
        sys.exit(f'No HSC calexp for tract {TRACT} patch {patch} band {band}')
    return cands[0]


def main():
    a = parse_args()
    band, patch = a.filter, a.patch
    suffix = patch          # output-file key = patch (filter is in the instrument dir);
                            #  matches 56_single's band=args.tile convention
    inst = f'hsc_{band}'
    out = WORK / inst / patch / 'psf'
    out.mkdir(parents=True, exist_ok=True)

    calexp = find_calexp(patch, band)
    # SExtractor's CFITSIO chokes on the LSST calexp's exotic binary-table
    # extensions ("Malformed FITS binary-table header"), so extract clean
    # single-HDU IMAGE + VARIANCE (astropy reads the calexp fine), keeping the
    # IMAGE WCS header.  MASK stays in-memory (used only by the pixel tests).
    with fits.open(calexp) as h:
        sci   = h[1].data.astype(np.float32)     # IMAGE
        mask  = h[2].data.astype(np.int32)       # MASK
        imhdr = h[1].header                      # standard FITS WCS lives here
    img_clean = out / f'image_{suffix}.fits'
    fits.PrimaryHDU(sci, header=imhdr).writeto(img_clean, overwrite=True)
    sex_img = str(img_clean)
    zp = ZP_NOMINAL
    # WEIGHT_TYPE NONE: self-estimated noise.  The LSST VARIANCE plane (with inf
    # in NO_DATA) degenerates SExtractor's MAP_VAR error model (SNR_WIN→1e30,
    # CLASS_STAR→0), so we let SExtractor estimate the background RMS itself —
    # validated to give a clean stellar locus (3.38 px) + sensible SNR/CLASS_STAR.
    # The MASK plane still drives the saturation/gap rejection below.

    print(f'Instrument : {inst}  (HSC SSP s23b deepCoadd)')
    print(f'Patch/band : {TRACT}/{patch} / {band.upper()}   ZP_AB={zp:.4f} (provisional)')
    print(f'Source     : {calexp.name}')
    print(f'             extracted IMAGE[1]; WEIGHT NONE; MASK[2]-bit saturation; SNR>{a.snr_min}')

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
    # ── saturation = LSST MASK SAT/BAD bits in the stellar core ──
    half = a.core_box // 2
    xi = np.clip(np.round(xx).astype(int), half, nx-half-1)
    yi = np.clip(np.round(yy).astype(int), half, ny-half-1)
    saturated = np.zeros(len(obj), bool)
    for k in range(len(obj)):
        box = mask[yi[k]-half:yi[k]+half+1, xi[k]-half:xi[k]+half+1]
        saturated[k] = bool(np.any(box & CORE_REJECT))
    is_pt = (cs > 0.8) & (snr > a.snr_min)
    sat_pt = saturated & is_pt & np.isfinite(mag)
    onset_mag = float(np.percentile(mag[sat_pt], 90)) if sat_pt.sum() >= 5 else None
    # bright limit = the MEASURED saturation magnitude: veto everything brighter
    # than the onset (uses the measurement, same convention as the other
    # missions — no ad-hoc magnitude cut).
    if onset_mag is not None and np.isfinite(onset_mag):
        saturated = saturated | (np.isfinite(mag) & (mag < onset_mag))
    print(f'  saturation : {int(saturated.sum())} excluded  '
          f'(measured saturation mag = {onset_mag if onset_mag is None else round(onset_mag,2)})')

    # ── Gaia-DR3-anchored stellar locus (immune to compact-galaxy bias) ──
    gaia_star = gaia_star_mask(xx, yy, imhdr, radius_arcsec=0.6)
    locus_base = gaia_star & (flg < 2) & (fr > 0) & (snr > a.snr_min) & (~saturated)
    anchored = int(locus_base.sum()) >= 15
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
    onloc = locus_base & (np.abs(fr - med) < 3.0 * mad)
    psf_fwhm_est = (float(np.median(fwhm[onloc])) if onloc.sum() >= 3
                    else float(np.median(fwhm[locus_base])))
    fwhm_min = 0.5 * psf_fwhm_est
    std = float(np.std(fr[locus_base]))
    cut = med - 3.0 * mad
    upper = med + 3.0 * mad
    print(f'  locus anchor : {"Gaia DR3" if anchored else "CLASS_STAR (Gaia<15)"}'
          f'  ({int(gaia_star.sum())} Gaia matched, {int(locus_base.sum())} locus stars)')
    print(f'  PSF FWHM est = {psf_fwhm_est:.2f} px ; stellar locus = {med:.3f} px '
          f'(MAD-σ {mad:.3f}) ; artifact cut = {cut:.3f} px')

    # bright-end override window (a few mag below saturation)
    bright_pivot = ((onset_mag + 4.0) if (onset_mag is not None and np.isfinite(onset_mag))
                    else (float(np.percentile(mag[locus_base], 25)) if locus_base.sum() else 99.0))

    # gaps (VIGNET over NO_DATA/EDGE) + neighbour contamination
    edge = (flg & 8) > 0
    cheap = (cs > 0.8) & (snr > a.snr_min) & (elon < 1.5) & (fr > 0)
    xiv = np.clip(np.round(xx).astype(int), VIG_HALF, nx-VIG_HALF-1)
    yiv = np.clip(np.round(yy).astype(int), VIG_HALF, ny-VIG_HALF-1)
    gap_masked = np.zeros(len(obj), bool)
    for k in np.where(cheap)[0]:
        box = mask[yiv[k]-VIG_HALF:yiv[k]+VIG_HALF+1, xiv[k]-VIG_HALF:xiv[k]+VIG_HALF+1]
        gap_masked[k] = int(np.sum((box & GAP_REJECT) != 0)) > 5
    # neighbour contamination: reject when a DETECTED companion lies within the
    # PSF model stamp (~3.4") and is no more than DMAG fainter.  The old 1"
    # radius was far smaller than the 3.4" stamp, so companions at 1-3.4"
    # escaped it; aggressive SExtractor deblending (hsc_<b>.sex) recovers blended
    # companions so they appear here as catalog detections.
    tree = cKDTree(np.column_stack([xx, yy]))
    fwhm_px = 2.0 * med                    # PSF FWHM from the half-light locus
    R_nbr   = a.nbr_fwhm * fwhm_px         # isolation radius (px) = N x FWHM
    # STRICT isolation: reject if ANY detected object lies within R_nbr.  Galaxy-
    # dense COSMOS has faint neighbours everywhere, so this trades sample size for
    # cleanliness (no brightness gate — even faint galaxies disqualify a star).
    contaminated = np.zeros(len(obj), bool)
    cand = (cs > 0.8) & (snr > a.snr_min) & (fr > 0)
    for k in np.where(cand)[0]:
        for j in tree.query_ball_point([xx[k], yy[k]], R_nbr):
            if j != k:
                contaminated[k] = True; break
    # PSF model stamp is SMALLER than the isolation radius — the fit core is then
    # guaranteed clean even if a galaxy sits just outside the isolation circle.
    psf_size = int(round(2.0 * a.psf_fwhm * fwhm_px)) | 1
    e_round = ell < 0.20

    # scalar bright-end relaxed upper (used by the two-pass anchor override)
    upper_bright = med + 6.0 * mad

    # ── TILTED (magnitude-dependent) stellar locus ──────────────────────────
    # Ground PSF FLUX_RADIUS descends toward faint mags (faint stars lose their
    # low-SNR wings).  Fit a STRAIGHT line FR = slope·mag + icpt to the locus
    # stars across the full magnitude range — seeded by the constant median and
    # asymmetric-clipped (galaxies sit ABOVE the ridge) — so the artifact cut
    # and upper rail are PARALLEL tilted rails at locus(mag) ∓ 3·MAD.  Robust
    # fallbacks: too-few stars / a wild slope revert to the flat constant median.
    wpx   = max(0.40, 8.0 * mad)                    # px half-window around ridge
    tsamp = ((cs > 0.8) & (snr > a.snr_min) & (flg < 2) & (~saturated)
             & (fr > 0) & np.isfinite(mag) & (np.abs(fr - med) < wpx)) | locus_base
    slope, icpt, mad_t, tilted = 0.0, med, mad, False
    if int(tsamp.sum()) >= 20:
        tm, tf = mag[tsamp], fr[tsamp]
        kp, bb, ss = (np.isfinite(tm) & np.isfinite(tf)), (0.0, med), mad
        for _ in range(8):
            if kp.sum() < 8:
                break
            bb = np.polyfit(tm[kp], tf[kp], 1)
            rr = tf - np.polyval(bb, tm)
            ss = 1.4826 * float(np.median(np.abs(rr[kp] - np.median(rr[kp]))))
            if ss <= 0:
                break
            kp = (rr > -3.5 * ss) & (rr < 3.0 * ss)     # tighter above (galaxies)
        slope = float(np.clip(bb[0], -0.08, 0.02))      # sane ground slope px/mag
        icpt, mad_t, tilted = float(bb[1]), max(float(ss), 0.030), True
    locus_of = lambda mm: slope * np.asarray(mm, float) + icpt

    # FR band: parallel tilted rails (bright end relaxed for the anchor override)
    cut_arr   = locus_of(mag) - 3.0 * mad_t
    upper_arr = np.where(np.isfinite(mag) & (mag < bright_pivot),
                         locus_of(mag) + 6.0 * mad_t, locus_of(mag) + 3.0 * mad_t)
    keep_fr   = (fr > cut_arr) & (fr < upper_arr)
    print(f'  tilted locus : {"on" if tilted else "OFF(const)"}  '
          f'FR={slope:+.4f}·mag{icpt:+.3f}px  ({slope*PIXSCALE*1000:+.1f} mas/mag)  '
          f'MAD_t={mad_t:.4f}px')
    # NO ad-hoc magnitude cut: bright end = measured saturation mag (above),
    # faint end = S/N floor + the FLUX_RADIUS locus band (rejects the galaxy sea)
    star = ((cs > 0.8) & (snr > a.snr_min) & (elon < 1.5) & (fwhm > fwhm_min)
            & keep_fr & e_round
            & (~saturated) & (~edge) & (~contaminated) & (~gap_masked))
    print(f'  PSF stars selected : {star.sum()}   '
          f'(bright=sat mag, faint=SNR>{a.snr_min:.0f}+locus; saturated/bad excl '
          f'{saturated.sum()}, gap {gap_masked.sum()}, '
          f'isolation (any nbr <{a.nbr_fwhm:.1f}xFWHM={R_nbr*PIXSCALE:.2f}\") '
          f'{int(contaminated.sum())}; PSF_SIZE={psf_size}px)')

    # ── two-pass PSFEx: validate, then rebuild from accepted∪bright-override ──
    sel_idx = np.where(star)[0]
    all_cat = out / f'stars_all_{suffix}.fits'
    hcat[2].data = obj[sel_idx]; hcat.writeto(all_cat, overwrite=True)
    fwhm_lo = max(1.2, 0.6 * psf_fwhm_est); fwhm_hi = 2.5 * psf_fwhm_est
    print(f'\n── PSFEx pass 1 (validate; SAMPLE_FWHMRANGE {fwhm_lo:.2f},{fwhm_hi:.2f}) ──', flush=True)
    p1out = out / f'p1outcat_{suffix}.fits'
    r = subprocess.run(['psfex', str(all_cat), '-c', str(CONFIGS / 'psfex_hsc.psfex'),
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

    # PSF_ACCURACY 0.5 only when there are bright shape-override anchors to
    # protect from PSFEx's chi2-cleaning (the NISP case).  HSC saturates early
    # → 0 bright anchors, so use the normal tight 0.01: a proper fit WITH
    # PSFEx's standard cleaning of the noisy faint stars (no over-smoothing).
    acc2 = '0.5' if n_ovr > 0 else '0.01'
    print(f'── PSFEx pass 2 (final model; PSF_ACCURACY {acc2}) ──', flush=True)
    cmd = ['psfex', str(star_cat), '-c', str(CONFIGS / 'psfex_hsc.psfex'),
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

    # ── 5. meta JSON (keys mirror the COSMOS-Web builders for 56_single) ──
    meta = {
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'instrument': inst, 'tile': patch, 'filter': band, 'channel': 'hsc',
        'zp_ab': float(zp), 'zp_provisional': True, 'pixscale_arcsec': PIXSCALE,
        'sci_source_file': str(calexp), 'sci_source_ext': 1,
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
        'locus_anchor': ('gaia_dr3' if anchored else 'class_star'),
        'n_gaia_matched': int(gaia_star.sum()),
        'sat_onset_mag': onset_mag,
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
        'recipe': 'HSC LSST calexp IMAGE[1]; WEIGHT NONE; MASK-bit saturation; '
                  'Gaia-anchored locus; two-pass bright-anchor PSF',
    }
    (out / f'psf_{suffix}.meta.json').write_text(json.dumps(meta, indent=2))
    print(f'\n[save] {out / f"psf_{band}.meta.json"}')
    print('=== HSC PSF model build complete ===')


if __name__ == '__main__':
    main()
