#!/usr/bin/env python
"""
54_step3a_build_psf_model_euclid.py — Euclid VIS (DR1 MER) PSF model
builder.  Single-HDU adaptation of the JWST builder, carrying the SAME
"good star" technology that was validated on the JWST bands:

  - FWHM artifact floor (kills the single-pixel CR/hot-pixel "second
    locus" that otherwise inflates the MAD and drives the artifact cut
    to negative half-light radius)
  - dynamic PSFEx SAMPLE_FWHMRANGE = [0.6,2.5]×PSF_FWHM so the
    acceptance window never clips the stellar locus / bright stars
  - VIGNET gap-mask test (a star whose VIGNET overlaps a coverage hole
    is rejected by PSFEx; pre-flag it so it never poses as a PSF star)
  - ELLIPTICITY<0.20 candidate ceiling (PSFEx SAMPLE_MAXELLIP=0.18)

Format differences from JWST:
  - single primary HDU (no SCI/WHT split); WEIGHT_TYPE NONE
  - ZP_AB from the MAGZERO header keyword
  - pixel scale 0.10"/px; VIGNET 51 px (VIS PSF ≈ 1.6 px FWHM)
  - tile id = the 9-digit number in EUC_MER_BGSUB-MOSAIC-VIS_TILE<id>

Star selection (identical philosophy to JWST):
  CLASS_STAR>0.8, SNR_WIN>100, ELONGATION<1.5, ELLIPTICITY<0.20,
  FWHM>0.5·PSF_FWHM,  locus-3·MAD < FLUX_RADIUS < locus+2.5σ,
  masked_core==0,  not gap-masked,  NOT (n_1>=2 AND FLAGS<2).
  Diffraction-spike (FLAGS>=2) stars are KEPT.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree
from psf_saturation import core_peak, detect_saturation

PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS  = PROJECT / 'configs'
EUC_ROOT = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')
WORK     = Path('/Volumes/exdisk1/data/photometry_v04')

PIX_VIS  = 0.10   # arcsec / pix — VIS *and* NISP (MER resamples NISP onto VIS grid)
VIG_HALF = 25     # VIGNET(51,51) half-size, for the gap-mask test

# band key -> (MER mosaic product token, instrument key, detection conv filter)
# NISP Y/J/H are the same single-HDU MER mosaics as VIS (10200², 0.1"/px,
# MAGZERO in header), so the ONLY differences are the product token + config.
BAND_MAP = {
    'vis': ('VIS',   'euclid_vis',     'gauss_2.0_5x5.conv', 'psfex_euclid_vis.psfex'),
    'y':   ('NIR-Y', 'euclid_nisp_y',  'gauss_4.0_7x7.conv', 'psfex_euclid_nisp.psfex'),
    'j':   ('NIR-J', 'euclid_nisp_j',  'gauss_4.0_7x7.conv', 'psfex_euclid_nisp.psfex'),
    'h':   ('NIR-H', 'euclid_nisp_h',  'gauss_4.0_7x7.conv', 'psfex_euclid_nisp.psfex'),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--band', default='vis', choices=list(BAND_MAP),
                   help='vis | y | j | h  (NISP = y/j/h)')
    p.add_argument('--instrument', default=None,
                   help='output key; defaults from --band (euclid_vis / euclid_nisp_<b>)')
    p.add_argument('--tile', required=True,
                   help='Euclid tile id (the 9-digit number from TILE<id>)')
    p.add_argument('--core-box', type=int, default=3,
                   help='central NxN box to test for masked (=0) pixels')
    p.add_argument('--reuse-pass1', action='store_true')
    return p.parse_args()


def find_tile_image(tile_id: str, product: str) -> Path:
    matches = list(EUC_ROOT.glob(
        f'EUC_MER_BGSUB-MOSAIC-{product}_TILE{tile_id}*.fits'))
    matches = [m for m in matches if not m.name.startswith('._')]
    if not matches:
        sys.exit(f'No Euclid {product} image matches TILE{tile_id} in {EUC_ROOT}')
    return matches[0]


GAIA_CSV = Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS/gaia_dr3_cosmos.csv')


def gaia_star_mask(xx, yy, hdr, radius_arcsec=0.6):
    """Boolean mask: detections matched to a Gaia DR3 source within `radius`.

    COSMOS is galaxy-rich, so CLASS_STAR>0.8 admits unresolved compact galaxies
    (badly for the broad NISP PSF), biasing the FLUX_RADIUS stellar locus high
    and broadening its MAD.  Anchoring the locus to Gaia point sources gives the
    real, tight stellar locus.  Returns all-False on any failure so the caller
    falls back to the CLASS_STAR locus."""
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


def main():
    args = parse_args()
    product, inst_default, conv, psfexcfg = BAND_MAP[args.band]
    instrument = args.instrument or inst_default
    sexcfg = f'{instrument}.sex'
    img = find_tile_image(args.tile, product)
    out = WORK / instrument / args.tile / 'psf'
    out.mkdir(parents=True, exist_ok=True)

    # read ZP and SCI from the single HDU
    with fits.open(img) as h:
        sci = h[0].data.astype(np.float32)
        hdr = h[0].header
    zp = float(hdr['MAGZERO'])
    print(f'Instrument : {instrument}  (Euclid {product}, band {args.band})')
    print(f'Tile       : {args.tile}   ZP_AB={zp:.4f}   PIX={PIX_VIS}″')
    print(f'Image      : {img.name}  shape={sci.shape}')

    # SExtractor reads the original single-HDU MER mosaic DIRECTLY — no sci_
    # copy.  The in-memory `sci` (already loaded above) drives the masked-core
    # / saturation pixel work.
    cat1 = out / f'pass1_{args.tile}.fits'
    if not (args.reuse_pass1 and cat1.exists()):
        print('\n── SExtractor pass 1 ──', flush=True)
        cmd = ['sex', str(img),
               '-c', str(CONFIGS / sexcfg),
               '-CATALOG_NAME', str(cat1),
               '-PARAMETERS_NAME', str(CONFIGS / 'pass1_euclid_vis.param'),
               '-FILTER_NAME', str(CONFIGS / conv),
               '-STARNNW_NAME', str(CONFIGS / 'default.nnw'),
               '-MAG_ZEROPOINT', f'{zp:.4f}',
               '-WEIGHT_TYPE', 'NONE',
               '-VERBOSE_TYPE', 'QUIET']
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-1500:], file=sys.stderr); sys.exit('SExtractor failed')
        print(f'  wall {time.time()-t0:.1f}s')

    # ── star selection ──
    print('\n── star selection ──', flush=True)
    hcat = fits.open(cat1)
    obj  = hcat[2].data
    mag  = np.asarray(obj['MAG_AUTO'], float)
    fr   = np.asarray(obj['FLUX_RADIUS'], float)
    fwhm = np.asarray(obj['FWHM_IMAGE'], float)
    cs   = np.asarray(obj['CLASS_STAR'], float)
    snr  = np.asarray(obj['SNR_WIN'], float)
    flg  = np.asarray(obj['FLAGS'], int)
    elon = np.asarray(obj['ELONGATION'], float)
    ell_arr = np.asarray(obj['ELLIPTICITY'], float)
    xx   = np.asarray(obj['X_IMAGE'], float)
    yy   = np.asarray(obj['Y_IMAGE'], float)

    # Gaia match (anchors the stellar locus below to real stars).
    gaia_star = gaia_star_mask(xx, yy, hdr)

    # masked-core saturation test on SCI
    half = args.core_box // 2
    ny, nx = sci.shape
    xi = np.clip(np.round(xx).astype(int), half, nx-half-1)
    yi = np.clip(np.round(yy).astype(int), half, ny-half-1)
    masked_core = np.zeros(len(obj), int)
    for k in range(len(obj)):
        box = sci[yi[k]-half:yi[k]+half+1, xi[k]-half:xi[k]+half+1]
        masked_core[k] = np.sum(box == 0)
    edge = (flg & 8) > 0

    # Saturation = masked core (Euclid VIS MER zeroes saturated cores) OR a
    # core peak sitting on the bright-end PEAK PLATEAU (non-masking case).
    # detect_saturation finds the plateau from the point-source sample; for
    # Euclid the cores are masked so usually no plateau is found and the
    # masked-core test does the work (the onset magnitude is still reported
    # for the diagnostic line).
    is_point = (cs > 0.8) & (snr > 100)
    peak = core_peak(sci, xx, yy, half=2, idx=np.where(is_point)[0])
    sat_peak_level, onset_mag = detect_saturation(mag, peak, masked_core, is_point)
    peak_saturated = np.isfinite(peak) & (peak >= sat_peak_level)
    saturated = (masked_core > 0) | peak_saturated
    print(f'  saturation: masked-core={int((masked_core>0).sum())}  '
          f'peak-plateau={int(peak_saturated.sum())}  '
          f'(sat_peak_level={"∞" if not np.isfinite(sat_peak_level) else f"{sat_peak_level:.3g}"}, '
          f'onset_mag={onset_mag if onset_mag is None else round(onset_mag,2)})')

    # ── Gaia-anchored stellar locus ──
    # Use Gaia DR3 point sources (unsaturated, FLAGS<2, SNR>100) to define the
    # TRUE stellar locus — immune to the galaxy contamination that biases a
    # CLASS_STAR-based locus high (NISP-Y: 3.19 px contaminated → ~2.6 px Gaia).
    # Fall back to the CLASS_STAR base only if too few Gaia stars are matched.
    locus_base = gaia_star & (flg < 2) & (fr > 0) & (snr > 100) & (~saturated)
    anchored = int(locus_base.sum()) >= 15
    if not anchored:
        locus_base = ((cs > 0.8) & (snr > 100) & (flg < 2) & (elon < 1.5)
                      & (fr > 0) & (~saturated))
    med = float(np.median(fr[locus_base]))
    mad = 1.4826 * float(np.median(np.abs(fr[locus_base] - med)))
    for _ in range(20):                              # iterative 3·MAD clip
        sel = np.abs(fr[locus_base] - med) < 3.0 * mad
        if sel.sum() < 5:
            break
        nmed = float(np.median(fr[locus_base][sel]))
        nmad = 1.4826 * float(np.median(np.abs(fr[locus_base][sel] - nmed)))
        if abs(nmed - med) < 1e-4 and abs(nmad - mad) < 1e-4:
            med, mad = nmed, nmad; break
        med, mad = nmed, nmad
    onloc = locus_base & (np.abs(fr - med) < 3.0 * mad)
    psf_fwhm_est = (float(np.median(fwhm[onloc])) if onloc.sum() >= 3
                    else float(np.median(fwhm[locus_base])))
    fwhm_min = 0.5 * psf_fwhm_est
    std0 = float(np.std(fr[locus_base]))
    cut   = med - 3.0 * mad        # artifact floor — tight (close to the locus)
    upper = med + 3.0 * mad        # upper ceiling — rejects resolved galaxies
    print(f'  locus anchor : {"Gaia DR3" if anchored else "CLASS_STAR (Gaia<15)"}'
          f'  ({int(gaia_star.sum())} Gaia matched, {int(locus_base.sum())} locus stars)')
    print(f'  PSF FWHM est : {psf_fwhm_est:.2f} px   (FWHM-min {fwhm_min:.2f} px)')

    # Bright limit = the RECORDED saturation magnitude (peak-counts plateau,
    # onset_mag) — the same convention as HST/Euclid-VIS.  Stars brighter than
    # the onset (saturating / FR-hooked cores) are vetoed; every good star
    # FAINTER than it is kept.  `saturated` already flags peak>=sat_peak_level.
    if onset_mag is not None and np.isfinite(onset_mag):
        bright_sat = np.isfinite(mag) & (mag < onset_mag)
        saturated = saturated | bright_sat
        print(f'  bright limit : saturation onset {onset_mag:.2f} mag  '
              f'({int(bright_sat.sum())} brighter vetoed; fainter good stars kept)')
    else:
        print('  bright limit : peak-plateau veto only (no onset detected)')
    # Bright-end PSF stars (mag < bright_pivot) out-resolve the empirical NISP
    # PSF model (~1-2% floor) and are χ²-flagged by PSFEx even though their
    # shape is stellar.  The QA/acceptance step reinstates the shape-good ones
    # (FR on locus + FWHM consistent + low ellipticity); record the pivot here.
    bright_pivot = (onset_mag + 4.0) if (onset_mag is not None and np.isfinite(onset_mag)) else 18.0

    # VIGNET-gap test: a star whose VIGNET footprint overlaps a coverage
    # hole (zero pixels) is surprise-rejected by PSFEx; pre-flag it.
    xiv = np.clip(np.round(xx).astype(int), VIG_HALF, nx-VIG_HALF-1)
    yiv = np.clip(np.round(yy).astype(int), VIG_HALF, ny-VIG_HALF-1)
    cheap = (cs > 0.8) & (snr > 100) & (elon < 1.5) & (fr > 0)
    vignet_masked = np.zeros(len(obj), int)
    for k in np.where(cheap)[0]:
        box = sci[yiv[k]-VIG_HALF:yiv[k]+VIG_HALF+1, xiv[k]-VIG_HALF:xiv[k]+VIG_HALF+1]
        vignet_masked[k] = np.sum(box == 0)
    gap_masked = vignet_masked > 5

    # n_1 (1″ at 0.1″/px → 10 px)
    tree = cKDTree(np.column_stack([xx, yy]))
    r_pix = 1.0 / PIX_VIS
    n1 = np.array([len(tree.query_ball_point([xx[k], yy[k]], r_pix)) - 1
                   for k in range(len(obj))])
    contaminated = (n1 >= 2) & (flg < 2)

    # ELLIPTICITY<0.20 ceiling + TIGHT FR band [locus−3·MAD, locus+3·MAD]
    # around the Gaia-anchored locus.  The tight upper bound rejects the
    # resolved/compact galaxies that CLASS_STAR lets through.
    e_round = ell_arr < 0.20
    # FR band — DIFFERENT criterion bright vs faint: tight [cut, upper=med+3·MAD]
    # at the faint end (rejects compact galaxies that CLASS_STAR lets through);
    # a relaxed upper (med+6·MAD) for the bright end (mag<bright_pivot), where
    # the real stellar locus sits a touch higher/wider but still well below the
    # saturation hook.  Floor = the tight artifact cut everywhere; faint good
    # stars are NOT lost.
    upper_bright = med + 6.0 * mad
    upper_arr = np.where(np.isfinite(mag) & (mag < bright_pivot), upper_bright, upper)
    keep_fr = (fr > cut) & (fr < upper_arr)
    pre = ((cs > 0.8) & (snr > 100) & (elon < 1.5) & (fwhm > fwhm_min)
           & keep_fr & e_round & (~saturated) & (~edge))
    star = pre & (~contaminated) & (~gap_masked)
    dropped_by_gap = pre & gap_masked
    dropped_by_contam = pre & (~gap_masked) & contaminated
    print(f'  stellar locus = {med:.3f} px = {med*PIX_VIS:.3f}"   '
          f'(MAD-σ {mad:.3f} px = {mad*PIX_VIS:.4f}")')
    print(f'  artifact cut  = locus−3·MAD = {cut:.3f} px = {cut*PIX_VIS:.3f}"   '
          f'upper = locus+3·MAD = {upper:.3f} px')
    print(f'  PSF stars selected : {int(star.sum())}'
          f'  (saturated {int(saturated.sum())}, gap {int(dropped_by_gap.sum())}, '
          f'contam {int(dropped_by_contam.sum())})')
    if star.sum():
        print(f'  min FR among PSF stars            : {fr[star].min():.3f} px'
              f'  ({"OK" if fr[star].min() > cut else "FAIL"} > cut {cut:.3f})')

    # write filtered LDAC
    star_cat = out / f'stars_{args.tile}.fits'
    hcat[2].data = obj[star]
    hcat.writeto(star_cat, overwrite=True)
    print(f'  wrote {star_cat.name} ({star.sum()} stars)')

    # ── PSFEx — dynamic SAMPLE_FWHMRANGE scaled to the measured PSF FWHM ──
    fwhm_lo = max(0.8, 0.6 * psf_fwhm_est)
    fwhm_hi = 2.5 * psf_fwhm_est
    print('\n── PSFEx ──', flush=True)
    print(f'  SAMPLE_FWHMRANGE = {fwhm_lo:.2f},{fwhm_hi:.2f} px '
          f'(scaled to PSF FWHM {psf_fwhm_est:.2f})')
    cmd = ['psfex', str(star_cat),
           '-c', str(CONFIGS / psfexcfg),
           '-SAMPLE_FWHMRANGE', f'{fwhm_lo:.2f},{fwhm_hi:.2f}',
           '-CHECKIMAGE_TYPE', 'RESIDUALS,PROTOTYPES,SNAPSHOTS,SAMPLES',
           '-CHECKIMAGE_NAME',
           f'{out}/resi.fits,{out}/proto.fits,{out}/snap.fits,{out}/samp.fits']
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], file=sys.stderr); sys.exit('PSFEx failed')
    psf = star_cat.with_suffix('.psf')
    if not psf.exists():
        alt = Path.cwd() / psf.name
        if alt.exists(): shutil.move(alt, psf)
    ph = fits.open(psf)[1].header
    print(f'  wall {time.time()-t0:.1f}s')
    print(f'  PSF model: chi2={ph.get("CHI2",-1):.3f}  '
          f'FWHM={ph.get("PSF_FWHM",-1):.2f} px  accepted={ph.get("ACCEPTED","?")}')

    # diagnostic JSON
    (out / f'psf_{args.tile}.meta.json').write_text(json.dumps({
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'instrument': instrument, 'band': args.band, 'tile': args.tile,
        'zp_ab': zp, 'pixel_scale_arcsec': PIX_VIS,
        'sci_source_file': str(img), 'sci_source_ext': 0,
        'psf_fwhm_est_px': float(psf_fwhm_est),
        'sample_fwhmrange': [float(fwhm_lo), float(fwhm_hi)],
        'stellar_locus_px': float(med), 'locus_std_px': float(std0),
        'locus_mad_sigma_px': float(mad),
        'stellar_locus_arcsec': float(med * PIX_VIS),
        'locus_mad_arcsec': float(mad * PIX_VIS),
        'artifact_cut_px': float(cut), 'upper_px': float(upper),
        'upper_bright_px': float(upper_bright),
        'bright_pivot_mag': float(bright_pivot),
        'locus_anchor': ('gaia_dr3' if anchored else 'class_star'),
        'n_gaia_matched': int(gaia_star.sum()),
        'n_psf_stars': int(star.sum()),
        'n_spike_stars_kept': int((star & (flg >= 2)).sum()),
        'n_saturated_excluded': int(saturated.sum()),
        'sat_peak_level': (None if not np.isfinite(sat_peak_level) else float(sat_peak_level)),
        'sat_onset_mag': onset_mag,
        'n_peak_saturated': int(peak_saturated.sum()),
        'n_gap_masked_excluded': int(dropped_by_gap.sum()),
        'n_contaminated_excluded': int(dropped_by_contam.sum()),
        'psf_chi2': float(ph.get('CHI2', -1)),
        'psf_fwhm_px': float(ph.get('PSF_FWHM', -1)),
        'psf_accepted': int(ph.get('ACCEPTED', -1)),
        'psf_model': str(psf),
    }, indent=2))
    print(f'\n[save] {out / f"psf_{args.tile}.meta.json"}')
    print('=== Euclid VIS PSF build complete ===')


if __name__ == '__main__':
    main()
