#!/usr/bin/env python
"""
53_step3a_dual_photometry.py — Step 3a-③ part 1: mission-generic forced
PSF photometry + saturation flags + DAO morphology.

Per (mission, tile), for every band:

  1. SExtractor DUAL-IMAGE forced photometry
       detection   = chi2_<tile>.fits (+ its coverage MAP_WEIGHT), with
                     EXACTLY the same cold/hot configs + thresholds as
                     51_ — so the source lists and NUMBER ordering
                     reproduce 51_'s passes and the join to the merged
                     spine is exact, no positional matching
       measurement = native band image (+ its weight where one exists)
       PSF model   = the validated Step 3a-① PSFEx model
                     (psf_model path from the tile's meta JSON)
       → FLUX_PSF / MAG_PSF / CHI2_PSF / SPREAD_MODEL per source per band
  2. Saturation flags (FLAG, NEVER DROP — the B3 lesson from v01):
       is_saturated_<band> =
         core-masked   (≥1 zero pixel in the 5×5 core; JWST i2d and
                        Euclid VIS MER zero saturated cores)
         OR peak-plateau (core peak ≥ sat_peak_level from the 3a-① meta;
                        HST ACS drz keeps saturated cores at full well)
         OR MAG_PSF < sat_onset_mag (3a-① meta, when present)
       Saturated rows KEEP their record; downstream uses unsaturated
       bands only.
  3. DAO detection-stage morphology (3b classifier inputs):
       DAOStarFinder(fwhm = empirical PSF FWHM, threshold = 4σ) on the
       native band image → sharpness / roundness1 / roundness2, matched
       to the spine by KDTree within 1×FWHM.

DAO PSF-FITTING photometry on the point-source pool (photutils
PSFPhotometry with the same PSFEx model via GriddedPSFModel → the
psf_phot_consistent flag) is the second half of 3a-③ and lives in 66_
— it is not needed by the Step 3b classifier.

Output:
  <mission>_chi2/<tile>/phot/cat_<band>_<mode>_psf.fits   raw per band+mode
  <mission>_chi2/<tile>/phot/phot_<tile>.fits             wide join on spine
  <mission>_chi2/<tile>/phot/phot_<tile>.meta.json        resume marker + QA
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, join, vstack

HERE = Path(__file__).resolve().parent

# import 51_ as a module (digit-leading filename) — single source of truth
# for MISSIONS, FWHM lookup and paths
_spec = importlib.util.spec_from_file_location(
    'chi2_detect', HERE / '51_step3a_chi2_detect.py')
chi2_detect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chi2_detect)

MISSIONS = chi2_detect.MISSIONS
WORK     = chi2_detect.WORK
CONFIGS  = chi2_detect.CONFIGS
JWST_DIR = chi2_detect.JWST_DIR
HST_DIR  = chi2_detect.HST_DIR
empirical_fwhm_arcsec = chi2_detect.empirical_fwhm_arcsec
_euclid_mosaic        = chi2_detect._euclid_mosaic
_meta_path            = chi2_detect._meta_path

MODES = ['cold', 'hot']

# columns carried per band into the wide table (prefixed <band>_)
KEEP = ['FLUX_PSF', 'FLUXERR_PSF', 'MAG_PSF', 'MAGERR_PSF', 'CHI2_PSF',
        'SPREAD_MODEL', 'SPREADERR_MODEL', 'FLUX_AUTO', 'FLUXERR_AUTO',
        'MAG_AUTO', 'FLUX_RADIUS', 'FWHM_IMAGE', 'ELLIPTICITY',
        'CLASS_STAR', 'SNR_WIN', 'FLAGS']


def step(msg: str):
    print(f'\n── {msg} ──', flush=True)


# ----------------------------------------------------------------------
# per-mission measurement-image resolver
# ----------------------------------------------------------------------
def band_measurement(mission: str, tile: str, band: str):
    """Return (sex_image_arg, sex_weight_arg|None, sci_loader) for a band.
    sci_loader() lazily memmaps the SCI array for core-pixel tests."""
    if mission == 'jwst':
        p = JWST_DIR / (f'mosaic_nircam_{band}_COSMOS-Web_30mas_'
                        f'{tile}_v1.0_i2d.fits')
        if not p.exists():
            return None, None, None
        return (f'{p}[1]', f'{p}[4]',
                lambda: fits.open(p, memmap=True)['SCI'].data)
    if mission == 'hst':
        m = sorted(HST_DIR.glob(
            f'mosaic_cosmos_web_*_30mas_tile_{tile}_hst_acs_wfc_'
            f'f814w_drz.fits'))
        if not m:
            return None, None, None
        drz = m[0]
        wht = Path(str(drz).replace('_drz.fits', '_wht.fits'))
        return (str(drz), str(wht) if wht.exists() else None,
                lambda: fits.open(drz, memmap=True)[0].data)
    if mission == 'euclid':
        p = _euclid_mosaic(tile, band)
        if p is None:
            return None, None, None
        return str(p), None, lambda: fits.open(p, memmap=True)[0].data
    sys.exit(f'unknown mission {mission}')


def band_meta(mission: str, tile: str, band: str) -> dict:
    inst = MISSIONS[mission]['inst'][band]
    p = _meta_path(inst, tile, band)
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


# ----------------------------------------------------------------------
# dual-image SExtractor pass
# ----------------------------------------------------------------------
def run_dual(mission: str, tile: str, band: str, mode: str,
             meas_img: str, meas_wht: str | None, psf: str | None,
             zp: float, out_cat: Path) -> None:
    spec = MISSIONS[mission]
    work = WORK / f'{mission}_chi2' / tile
    det = work / f'chi2_{tile}.fits'
    det_wht = work / f'chi2_{tile}.wht.fits'
    seeing = empirical_fwhm_arcsec(mission, tile, spec['target'])[0]
    wt = ['MAP_WEIGHT', 'MAP_WEIGHT'] if meas_wht else ['MAP_WEIGHT', 'NONE']
    wi = [str(det_wht)] + ([meas_wht] if meas_wht else [])
    cmd = ['sex', f'{det},{meas_img}',
           '-c',               str(CONFIGS / f'chi2_{mode}.sex'),
           '-CATALOG_NAME',    str(out_cat),
           # pass2_psf.param = default_pass2 minus SPREAD_MODEL: the
           # galaxy-model machinery behind SPREAD_MODEL is ~170× the
           # cost of the entire rest of the pass (404.6 s vs 2.4 s on a
           # 4.2 Mpx benchmark, 2026-06-11) and 3b does not use it.
           # SPREAD_MODEL for the 66_ point-source pool: compute later,
           # pool candidates only, if the tilted locus + DAO vetoes
           # prove insufficient.
           '-PARAMETERS_NAME', str(CONFIGS / 'pass2_psf.param'),
           '-STARNNW_NAME',    str(CONFIGS / 'default.nnw'),
           '-FILTER_NAME',     str(CONFIGS / (
               'tophat_9.0_9x9.conv' if mode == 'cold'
               else 'gauss_3.0_5x5.conv')),
           '-WEIGHT_TYPE',     ','.join(wt),
           '-WEIGHT_IMAGE',    ','.join(wi),
           '-MAG_ZEROPOINT',   f'{zp:.4f}',
           '-PIXEL_SCALE',     f"{spec['pixscale']:.4f}",
           '-SEEING_FWHM',     f'{seeing:.4f}',
           '-CHECKIMAGE_TYPE', 'NONE',
           ]
    if mode == 'hot' and spec['hot_thresh'] is not None:
        cmd += ['-DETECT_THRESH', f"{spec['hot_thresh']:.2f}"]
    if psf:
        cmd += ['-PSF_NAME', psf]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr[-2000:], file=sys.stderr)
        sys.exit(f'dual {band}/{mode} failed (rc={r.returncode})')
    print(f'    dual {band}/{mode}: {time.time()-t0:.0f}s', flush=True)


# ----------------------------------------------------------------------
# saturation flags (flag, never drop)
# ----------------------------------------------------------------------
def saturation_flags(t: Table, sci_loader, meta: dict,
                     mag_col: str) -> tuple[np.ndarray, dict]:
    """is_saturated per row: core-masked OR peak-plateau OR onset-mag."""
    n = len(t)
    sat = np.zeros(n, dtype=bool)
    info = dict(sat_peak_level=meta.get('sat_peak_level'),
                sat_onset_mag=meta.get('sat_onset_mag'))
    onset = meta.get('sat_onset_mag')
    mag = np.asarray(t[mag_col], dtype=float)
    if onset is not None:
        sat |= (mag < onset) & (mag > -99)
    # pixel-level core tests only for plausibly bright sources
    bright = mag < ((onset + 1.5) if onset is not None else 21.0)
    idx = np.flatnonzero(bright & (mag > -99))
    info['n_bright_core_tested'] = int(len(idx))
    if len(idx):
        sci = sci_loader()
        ny, nx = sci.shape
        xx = np.asarray(t['X_IMAGE'], dtype=float) - 1
        yy = np.asarray(t['Y_IMAGE'], dtype=float) - 1
        peak_level = meta.get('sat_peak_level')
        for k in idx:
            xi, yi = int(round(xx[k])), int(round(yy[k]))
            if not (2 <= xi < nx-2 and 2 <= yi < ny-2):
                continue
            core = sci[yi-2:yi+3, xi-2:xi+3]
            if np.any(core == 0):                      # masked core
                sat[k] = True
            elif peak_level and np.nanmax(core) >= peak_level:
                sat[k] = True                          # peak plateau
        del sci
    info['n_saturated'] = int(sat.sum())
    return sat, info


# ----------------------------------------------------------------------
# DAO detection-stage morphology
# ----------------------------------------------------------------------
def dao_morphology(spine_xy: np.ndarray, sci_loader, fwhm_px: float
                   ) -> tuple[np.ndarray, dict]:
    """DAOStarFinder sharp/rnd1/rnd2 matched to spine positions."""
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    from scipy.spatial import cKDTree

    sci = np.asarray(sci_loader(), dtype=np.float32)
    _, med, std = sigma_clipped_stats(sci[::8, ::8], sigma=3.0, maxiters=3)
    t0 = time.time()
    finder = DAOStarFinder(fwhm=max(fwhm_px, 1.5), threshold=4.0*std,
                           exclude_border=True)
    found = finder(sci - med)
    del sci
    out = np.full((len(spine_xy), 3), np.nan, dtype=np.float32)
    info = dict(dao_bkg_std=float(std), dao_n_found=0, dao_n_matched=0,
                dao_s=round(time.time()-t0, 1))
    if found is None or len(found) == 0:
        return out, info
    info['dao_n_found'] = len(found)
    xy = np.column_stack([found['xcentroid'], found['ycentroid']])
    tree = cKDTree(xy)
    d, j = tree.query(spine_xy, k=1)
    ok = d <= max(fwhm_px, 1.5)
    out[ok, 0] = np.asarray(found['sharpness'], dtype=np.float32)[j[ok]]
    out[ok, 1] = np.asarray(found['roundness1'], dtype=np.float32)[j[ok]]
    out[ok, 2] = np.asarray(found['roundness2'], dtype=np.float32)[j[ok]]
    info['dao_n_matched'] = int(ok.sum())
    return out, info


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mission', required=True, choices=sorted(MISSIONS))
    p.add_argument('--tile', required=True)
    p.add_argument('--bands', nargs='+', default=None,
                   help='subset of the mission bands (default: all)')
    p.add_argument('--force', action='store_true')
    p.add_argument('--skip-dao', action='store_true',
                   help='skip DAO morphology (debug)')
    return p.parse_args()


def main():
    args = parse_args()
    spec = MISSIONS[args.mission]
    bands = args.bands or spec['bands']
    work = WORK / f'{args.mission}_chi2' / args.tile
    phot = work / 'phot'
    phot.mkdir(parents=True, exist_ok=True)
    out_path = phot / f'phot_{args.tile}.fits'
    meta_path = phot / f'phot_{args.tile}.meta.json'

    if out_path.exists() and meta_path.exists() and not args.force:
        print(f'{args.mission}/{args.tile}: phot already done — '
              f'--force to redo')
        return

    merged_path = work / f'merged_{args.tile}.fits'
    if not merged_path.exists():
        sys.exit(f'need 51_ first: {merged_path} missing')
    spine = Table.read(merged_path)
    print(f'mission {args.mission}  tile {args.tile}  '
          f'spine {len(spine):,} sources  bands {bands}')
    t_start = time.time()

    qa: dict = dict(mission=args.mission, tile=args.tile,
                    n_spine=len(spine), bands={})
    out = spine.copy()
    spine_xy = np.column_stack([np.asarray(spine['X_IMAGE']) - 1,
                                np.asarray(spine['Y_IMAGE']) - 1])

    for band in bands:
        step(f'band {band}')
        meas_img, meas_wht, sci_loader = band_measurement(
            args.mission, args.tile, band)
        if meas_img is None:
            print(f'  {band}: no imaging — skipped')
            qa['bands'][band] = dict(used=False)
            continue
        bmeta = band_meta(args.mission, args.tile, band)
        zp = float(bmeta.get('zp_ab', 30.0))
        psf = bmeta.get('psf_model')
        if psf:
            # prefer the 6×FWHM photometry crop (68_) — the full 201–301
            # sample morphology raster makes the per-source fit ~10–40×
            # slower for identical core photometry
            crop = Path(psf).with_name(Path(psf).stem + '_phot.psf')
            if crop.exists():
                psf = str(crop)
            elif not Path(psf).exists():
                print(f'  [warn] psf model missing on disk: {psf}')
                psf = None
        binfo = dict(used=True, zp_ab=zp, psf_model=psf,
                     psf_meta_found=bool(bmeta))

        # 1. dual-image forced photometry, cold + hot
        per_mode = []
        for mode in MODES:
            cat = phot / f'cat_{band}_{mode}_psf.fits'
            if not (cat.exists() and not args.force):
                run_dual(args.mission, args.tile, band, mode,
                         meas_img, meas_wht, psf, zp, cat)
            t = Table(fits.open(cat)[2].data)
            t['DETECT_MODE'] = np.full(len(t), mode)
            per_mode.append(t)
        bcat = vstack(per_mode, metadata_conflicts='silent')

        # 2. saturation flags on the band catalog
        sat, satinfo = saturation_flags(bcat, sci_loader, bmeta, 'MAG_PSF')
        bcat['IS_SATURATED'] = sat
        binfo.update(satinfo)

        # slim + prefix + join onto spine by (DETECT_MODE, NUMBER)
        keep = ['NUMBER', 'DETECT_MODE', 'IS_SATURATED'] + [
            c for c in KEEP if c in bcat.colnames]
        bcat = bcat[keep]
        for c in keep:
            if c not in ('NUMBER', 'DETECT_MODE'):
                bcat.rename_column(c, f'{band}_{c}')
        n_before = len(out)
        out = join(out, bcat, keys=['DETECT_MODE', 'NUMBER'],
                   join_type='left')
        if len(out) != n_before:
            sys.exit(f'{band}: join changed row count '
                     f'{n_before} → {len(out)} — NUMBER alignment broken')

        # 3. DAO morphology matched to the spine
        if not args.skip_dao:
            fwhm_px = (empirical_fwhm_arcsec(args.mission, args.tile,
                                             band)[0] / spec['pixscale'])
            dao, daoinfo = dao_morphology(spine_xy, sci_loader, fwhm_px)
            out[f'{band}_DAO_SHARP'] = dao[:, 0]
            out[f'{band}_DAO_RND1'] = dao[:, 1]
            out[f'{band}_DAO_RND2'] = dao[:, 2]
            binfo.update(daoinfo)
        qa['bands'][band] = binfo
        n_phot = int(np.isfinite(
            np.asarray(out[f'{band}_MAG_PSF'], dtype=float)).sum()) \
            if f'{band}_MAG_PSF' in out.colnames else 0
        print(f'  {band}: PSF mags {n_phot:,}/{len(out):,}  '
              f'saturated {binfo.get("n_saturated", 0):,}  '
              f'DAO matched {binfo.get("dao_n_matched", 0):,}')

    out.write(out_path, overwrite=True)
    qa.update(n_rows=len(out), n_cols=len(out.colnames),
              total_s=round(time.time()-t_start, 1),
              created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    meta_path.write_text(json.dumps(qa, indent=2))
    print(f'\n=== {args.mission}/{args.tile} photometry complete ===')
    print(f'  [save] {out_path}  ({len(out):,} rows × {len(out.colnames)} cols, '
          f'{time.time()-t_start:.0f}s)')


if __name__ == '__main__':
    main()
