#!/usr/bin/env python
"""
66_step3a_dao_psf_photometry.py — Step 3a-③ part 2: DAO (photutils)
PSF-fitting photometry on the POINT-SOURCE POOL, with the SAME PSFEx
model as the SExtractor pass, so SEx−DAO flux differences isolate the
fitting method → the psf_phot_consistent flag.

Pool (UNION — high recall; precision is Step 3b's job), per band:
  (a) any Gaia DR3 match (wide COSMOS CSV, 1″);
  (b) tilted stellar locus: FLUX_RADIUS within ±5·MAD of the
      Gaia-anchored locus (FR = slope·mag + intercept, the 3a-①
      ground recipe), MAG_PSF brighter than the locus purity limit;
  (c) DAO morphology vetoes: 0.2 < sharp < 1.0 and |rnd1|,|rnd2| < 1.
Saturated-in-band sources stay in the pool RECORD but are excluded
from fitting (flag, never drop).

PSF: the 3a-① PSFEx model (the 6×FWHM _phot crop from 68_) evaluated
on a --grid×--grid spatial grid via its own polynomial (POLDEG 1 or 3),
wrapped as a photutils GriddedPSFModel (oversampling = 1/PSF_SAMP = 2).

Fitting: photutils PSFPhotometry with SourceGrouper (ALLSTAR-style
simultaneous fits of overlapping stars), init positions/fluxes from the
53_ catalog, fit_shape ≈ 4×FWHM.

Output:
  <mission>_chi2/<tile>/phot/dao_<tile>.fits      pool rows × per-band
      DAO_FLUX/DAO_MAG/err, group size, qfit, in_pool_*, consistency
  <mission>_chi2/<tile>/phot/dao_<tile>.meta.json counts + Δmag stats
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    'chi2_detect', HERE / '51_step3a_chi2_detect.py')
chi2_detect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chi2_detect)
MISSIONS = chi2_detect.MISSIONS
WORK = chi2_detect.WORK
empirical_fwhm_arcsec = chi2_detect.empirical_fwhm_arcsec

_spec53 = importlib.util.spec_from_file_location(
    'dual_phot', HERE / '53_step3a_dual_photometry.py')
dual_phot = importlib.util.module_from_spec(_spec53)
_spec53.loader.exec_module(dual_phot)
band_measurement = dual_phot.band_measurement
band_meta = dual_phot.band_meta

GAIA_CSV = Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS/'
                'gaia_dr3_cosmos_wide.csv')
CONSISTENT_DMAG = 0.10        # |MAG_PSF − DAO_MAG| for psf_phot_consistent


# ----------------------------------------------------------------------
# PSFEx → GriddedPSFModel
# ----------------------------------------------------------------------
def psfex_poly_terms(deg: int, X: float, Y: float) -> list[float]:
    """PSFEx polynomial term ordering: for dy in 0..deg, dx in
    0..deg-dy → X^dx · Y^dy  (x varies fastest)."""
    return [X**dx * Y**dy
            for dy in range(deg + 1) for dx in range(deg + 1 - dy)]


def load_gridded_psf(psf_path: str, nx_img: int, ny_img: int,
                     grid: int):
    """Evaluate a PSFEx .psf on a grid×grid spatial grid and wrap it as
    a photutils GriddedPSFModel."""
    from photutils.psf import GriddedPSFModel
    with fits.open(psf_path) as h:
        hdr = h[1].header
        basis = h[1].data['PSF_MASK'][0]      # (ncoeff, ny, nx)
    deg = int(hdr.get('POLDEG1', 0)) if hdr.get('POLNGRP', 0) else 0
    samp = float(hdr['PSF_SAMP'])
    over = int(round(1.0 / samp))
    zx = float(hdr.get('POLZERO1', nx_img / 2))
    sx = float(hdr.get('POLSCAL1', nx_img))
    zy = float(hdr.get('POLZERO2', ny_img / 2))
    sy = float(hdr.get('POLSCAL2', ny_img))

    n_exp = (deg + 1) * (deg + 2) // 2
    if basis.shape[0] < n_exp:                # defensive: const model
        deg, n_exp = 0, 1
    xs = np.linspace(0.05 * nx_img, 0.95 * nx_img, grid)
    ys = np.linspace(0.05 * ny_img, 0.95 * ny_img, grid)
    stamps, xypos = [], []
    for y in ys:
        for x in xs:
            terms = psfex_poly_terms(deg, (x - zx) / sx, (y - zy) / sy)
            im = np.tensordot(np.array(terms[:basis.shape[0]]),
                              basis[:len(terms)], axes=1)
            im = np.clip(im, 0, None)
            tot = im.sum()
            # photutils oversampled-grid convention: unit TOTAL flux ⇒
            # the oversampled stamp sums to oversampling² (native-pixel
            # flux = sum of subpixels / over²).  Normalizing to 1 here
            # made fitted fluxes 4× too big (−1.5 mag offset, measured).
            if tot > 0:
                im = im * (over**2 / tot)
            stamps.append(im)
            xypos.append((float(x), float(y)))
    data = np.array(stamps, dtype=np.float64)
    nd = NDData(data, meta=dict(grid_xypos=xypos, oversampling=over))
    return GriddedPSFModel(nd), over


# ----------------------------------------------------------------------
# point-source pool
# ----------------------------------------------------------------------
def gaia_match_mask(t: Table, radius_arcsec: float = 1.0) -> np.ndarray:
    import pandas as pd
    from scipy.spatial import cKDTree
    g = pd.read_csv(GAIA_CSV, usecols=['ra', 'dec'])
    dec = np.asarray(t['DELTA_J2000'], dtype=float)
    cosd = np.cos(np.deg2rad(np.median(dec)))
    tree = cKDTree(np.column_stack([g['ra'].to_numpy() * cosd,
                                    g['dec'].to_numpy()]))
    d, _ = tree.query(np.column_stack(
        [np.asarray(t['ALPHA_J2000'], dtype=float) * cosd, dec]), k=1)
    return d * 3600.0 <= radius_arcsec


def tilted_locus_mask(mag: np.ndarray, fr: np.ndarray,
                      anchor: np.ndarray, nmad: float = 5.0
                      ) -> tuple[np.ndarray, dict]:
    """FR = slope·mag + icpt fit on the anchor stars; pool-in everything
    within ±nmad·MAD, brighter than the anchor faint end + 1 mag."""
    ok = anchor & np.isfinite(mag) & np.isfinite(fr) & (fr > 0)
    if ok.sum() < 8:                       # 3a-① Gaia-anchor threshold
        return np.zeros(len(mag), dtype=bool), dict(locus='too_few_anchors')
    slope, icpt = np.polyfit(mag[ok], fr[ok], 1)
    resid = fr[ok] - (slope * mag[ok] + icpt)
    mad = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 0.04)
    lim = float(np.nanmax(mag[ok])) + 1.0
    m = (np.isfinite(mag) & np.isfinite(fr) &
         (np.abs(fr - (slope * mag + icpt)) < nmad * mad) & (mag <= lim))
    return m, dict(locus_slope=round(float(slope), 5),
                   locus_icpt=round(float(icpt), 3),
                   locus_mad=round(float(mad), 4),
                   locus_mag_lim=round(lim, 2), n_anchors=int(ok.sum()))


def dao_veto_mask(t: Table, band: str) -> np.ndarray:
    sh = np.asarray(t[f'{band}_DAO_SHARP'], dtype=float)
    r1 = np.asarray(t[f'{band}_DAO_RND1'], dtype=float)
    r2 = np.asarray(t[f'{band}_DAO_RND2'], dtype=float)
    return ((sh > 0.2) & (sh < 1.0) &
            (np.abs(r1) < 1.0) & (np.abs(r2) < 1.0))


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mission', required=True, choices=sorted(MISSIONS))
    p.add_argument('--tile', required=True)
    p.add_argument('--bands', nargs='+', default=None)
    p.add_argument('--grid', type=int, default=3,
                   help='spatial grid for GriddedPSFModel (default 3×3)')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    warnings.filterwarnings('ignore')
    spec = MISSIONS[args.mission]
    bands = args.bands or spec['bands']
    phot_dir = WORK / f'{args.mission}_chi2' / args.tile / 'phot'
    phot_path = phot_dir / f'phot_{args.tile}.fits'
    out_path = phot_dir / f'dao_{args.tile}.fits'
    meta_path = phot_dir / f'dao_{args.tile}.meta.json'
    if out_path.exists() and meta_path.exists() and not args.force:
        print(f'{args.mission}/{args.tile}: dao already done — --force')
        return
    if not phot_path.exists():
        sys.exit(f'need 53_ first: {phot_path}')

    from photutils.psf import PSFPhotometry, SourceGrouper

    t = Table.read(phot_path)
    print(f'{args.mission}/{args.tile}: {len(t):,} sources')
    t0 = time.time()
    gaia = gaia_match_mask(t)
    qa: dict = dict(mission=args.mission, tile=args.tile,
                    n_sources=len(t), n_gaia=int(gaia.sum()), bands={})

    out = Table()
    out['SOURCE_ID'] = t['SOURCE_ID']
    pool_any = np.zeros(len(t), dtype=bool)

    for band in bands:
        if f'{band}_MAG_PSF' not in t.colnames:
            qa['bands'][band] = dict(used=False)
            continue
        step_t = time.time()
        mag = np.asarray(t[f'{band}_MAG_PSF'], dtype=float)
        mag[mag <= -99] = np.nan
        fr = np.asarray(t[f'{band}_FLUX_RADIUS'], dtype=float)
        sat = np.asarray(t[f'{band}_IS_SATURATED'], dtype=bool)
        binfo = {}

        locus, linfo = tilted_locus_mask(mag, fr, gaia & ~sat)
        binfo.update(linfo)
        dao_ok = dao_veto_mask(t, band)
        pool = gaia | locus | dao_ok
        binfo.update(n_pool=int(pool.sum()),
                     n_pool_gaia=int(gaia.sum()),
                     n_pool_locus=int(locus.sum()),
                     n_pool_daoveto=int(dao_ok.sum()))
        out[f'{band}_IN_POOL'] = pool
        pool_any |= pool
        fit = pool & ~sat & np.isfinite(mag)
        binfo['n_fit'] = int(fit.sum())

        bmeta = band_meta(args.mission, args.tile, band)
        zp = float(bmeta.get('zp_ab', 30.0))
        psf_path = bmeta.get('psf_model')
        crop = (Path(psf_path).with_name(Path(psf_path).stem + '_phot.psf')
                if psf_path else None)
        if crop and crop.exists():
            psf_path = str(crop)
        if not psf_path or not Path(psf_path).exists() or fit.sum() == 0:
            qa['bands'][band] = dict(binfo, used=False,
                                     reason='no psf model or empty pool')
            continue

        meas_img, _, sci_loader = band_measurement(args.mission,
                                                   args.tile, band)
        sci = np.asarray(sci_loader(), dtype=np.float64)
        ny, nx = sci.shape
        psf_model, over = load_gridded_psf(psf_path, nx, ny, args.grid)
        fwhm_px = (empirical_fwhm_arcsec(args.mission, args.tile, band)[0]
                   / spec['pixscale'])
        fshape = int(4 * fwhm_px) | 1
        grouper = SourceGrouper(min_separation=2.5 * fwhm_px)
        phot = PSFPhotometry(psf_model, fit_shape=(fshape, fshape),
                             grouper=grouper, aperture_radius=2 * fwhm_px)
        init = Table()
        init['x_init'] = np.asarray(t['X_IMAGE'], dtype=float)[fit] - 1
        init['y_init'] = np.asarray(t['Y_IMAGE'], dtype=float)[fit] - 1
        flux0 = np.asarray(t[f'{band}_FLUX_PSF'], dtype=float)[fit]
        init['flux_init'] = np.where(np.isfinite(flux0) & (flux0 > 0),
                                     flux0, 1.0)
        res = phot(sci, init_params=init)
        del sci

        n = len(t)
        dao_flux = np.full(n, np.nan)
        dao_err = np.full(n, np.nan)
        dao_mag = np.full(n, np.nan)
        grp = np.zeros(n, dtype=np.int32)
        qfit = np.full(n, np.nan)
        idx = np.flatnonzero(fit)
        dao_flux[idx] = np.asarray(res['flux_fit'], dtype=float)
        if 'flux_err' in res.colnames:
            dao_err[idx] = np.asarray(res['flux_err'], dtype=float)
        if 'group_size' in res.colnames:
            grp[idx] = np.asarray(res['group_size'], dtype=int)
        if 'qfit' in res.colnames:
            qfit[idx] = np.asarray(res['qfit'], dtype=float)
        pos = dao_flux > 0
        dao_mag[pos] = zp - 2.5 * np.log10(dao_flux[pos])
        out[f'{band}_DAO_FLUX'] = dao_flux
        out[f'{band}_DAO_FLUX_ERR'] = dao_err
        out[f'{band}_DAO_MAG'] = dao_mag
        out[f'{band}_DAO_GROUP_SIZE'] = grp
        out[f'{band}_DAO_QFIT'] = qfit

        # consistency is judged RELATIVE to the band's median offset:
        # the global SEx−DAO offset (wing truncation + normalization
        # conventions) is an aperture-correction term absorbed by the
        # Step 4 anchor calibration; the per-source flag isolates
        # method disagreement.
        dmag = dao_mag - mag
        good = np.isfinite(dmag)
        med = float(np.median(dmag[good])) if good.sum() else 0.0
        consistent = good & (np.abs(dmag - med) < CONSISTENT_DMAG)
        out[f'{band}_PSF_PHOT_CONSISTENT'] = consistent
        if good.sum():
            mad = float(1.4826 * np.median(np.abs(dmag[good] - med)))
            binfo.update(dmag_median=round(med, 4), dmag_mad=round(mad, 4),
                         n_consistent=int(consistent.sum()),
                         frac_consistent=round(consistent.sum() /
                                               good.sum(), 3))
        binfo.update(used=True, fit_shape=fshape,
                     band_s=round(time.time() - step_t, 1))
        qa['bands'][band] = binfo
        print(f"  {band}: pool {binfo['n_pool']:,} (gaia {gaia.sum()}, "
              f"locus {binfo['n_pool_locus']:,}, dao {binfo['n_pool_daoveto']:,})"
              f"  fit {binfo['n_fit']:,}  Δmag {binfo.get('dmag_median', 0):+.3f}"
              f"±{binfo.get('dmag_mad', 0):.3f}  consistent "
              f"{binfo.get('frac_consistent', 0):.0%}  "
              f"({binfo['band_s']:.0f}s)", flush=True)

    out = out[pool_any]
    out.write(out_path, overwrite=True)
    qa.update(n_pool_any=int(pool_any.sum()),
              total_s=round(time.time() - t0, 1),
              created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    meta_path.write_text(json.dumps(qa, indent=2))
    print(f'\n=== dao {args.mission}/{args.tile}: {len(out):,} pool rows '
          f'({time.time()-t0:.0f}s) ===\n  [save] {out_path}')


if __name__ == '__main__':
    main()
