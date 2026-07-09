#!/usr/bin/env python
"""
73_make_piff_panels.py — Piff versions of the two MODEL-DEPENDENT QA panels
(mag_vs_chi2 and psf_residuals) for the website toggle.

Only panels 2 and 5 depend on the PSF model; the other four (half-light,
saturation, samples, neighbour) are identical for PSFEx and Piff.  This
script fits Piff on the SAME v2-gated star list, then re-renders those two
panels with 56_'s exact drawing functions (injected Piff evaluator), plus a
PSFEx counterpart of panel 2 on the IDENTICAL raw-stamp chi2 estimator so the
toggle compares model vs model (not estimator vs estimator).

Writes into the tile's psf/ dir:
  mag_vs_chi2_psfex.png     raw-stamp chi2 vs PSFEx  (fair PSFEx state)
  mag_vs_chi2_piff.png      raw-stamp chi2 vs Piff
  psf_residuals_piff.png (+_thumb)   (data - Piff)/peak, same renderer

Usage:
  ./73_make_piff_panels.py --instrument hst_acs_f814w --tile A4
  ./73_make_piff_panels.py --instrument jwst_nircam_f444w --tile A4 --suffix f444w
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
WORK = Path('/Volumes/exdisk1/data/photometry_v04')


def _load(mod):
    spec = importlib.util.spec_from_file_location(mod, HERE / f'{mod}.py')
    m = importlib.util.module_from_spec(spec)
    sys.argv = [mod]                      # keep argparse quiet on import
    spec.loader.exec_module(m)
    return m


def raw_chi2(sci, xs, ys, psf_at, H):
    """Per-star reduced chi2 vs a model (amplitude+shift+bg lstsq; same
    estimator for both backends)."""
    from scipy.ndimage import shift as ndshift
    ny, nx = sci.shape
    yy, xx = np.mgrid[-H:H+1, -H:H+1]
    rad = np.hypot(xx, yy)
    fitm = rad < min(16, H - 1); outr = rad > min(14, 0.9 * H)
    out = np.full(len(xs), np.nan)
    for i, (X0, Y0) in enumerate(zip(xs, ys)):
        X, Y = X0 - 1.0, Y0 - 1.0
        xi, yi = int(round(X)), int(round(Y))
        if yi-H < 0 or xi-H < 0 or yi+H+1 > ny or xi+H+1 > nx:
            continue
        c = np.nan_to_num(sci[yi-H:yi+H+1, xi-H:xi+H+1].astype(float))
        P = psf_at(X, Y)
        if P.shape[0] != 2*H+1:
            hh = P.shape[0] // 2
            P = (P[hh-H:hh+H+1, hh-H:hh+H+1] if hh >= H else
                 np.pad(P, H-hh)[:2*H+1, :2*H+1])
        P = ndshift(P, (Y - yi, X - xi), order=3)
        gy, gx = np.gradient(P)
        co = np.linalg.lstsq(np.vstack([P[fitm], gx[fitm], gy[fitm],
                                        np.ones(fitm.sum())]).T,
                             c[fitm], rcond=None)[0]
        R = c - (co[0]*P + co[1]*gx + co[2]*gy + co[3])
        sig = 1.4826*np.median(np.abs(R[outr] - np.median(R[outr]))) + 1e-9
        out[i] = float(np.mean((R[fitm]/sig)**2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', required=True)
    ap.add_argument('--tile', required=True)
    ap.add_argument('--suffix', default=None)
    ap.add_argument('--stamp', type=int, default=None)
    args = ap.parse_args()
    suffix = args.suffix or args.tile
    psf_dir = WORK / args.instrument / args.tile / 'psf'
    meta = json.loads((psf_dir / f'psf_{suffix}.meta.json').read_text())
    # no_coverage stub: the ground builders (54_*_{ps1,lsdr10,sdss,unwise})
    # write a minimal meta and exit WITHOUT a PSF model when a tile has
    # <10 Gaia-anchored locus stars.  A stale .psf can linger and pull the
    # tile into the worklist, but there is no model to compare — skip
    # cleanly (mirrors 56_'s guard) instead of KeyError on sci_source_file.
    if meta.get('status') == 'no_coverage' or meta.get('n_model_stars', 1) == 0:
        print(f'  [skip] {args.instrument}/{args.tile}: no_coverage stub '
              f'(no PSF model — {meta.get("n_locus_stars", "?")} locus stars)')
        return
    scf, sce = meta['sci_source_file'], int(meta.get('sci_source_ext', 0))
    pixscale = float(meta.get('pixel_scale_arcsec')
                     or meta.get('pixscale_arcsec') or 0.030)

    qa = _load('56_make_psf_qa_plots_single')
    tp = _load('72_test_piff')
    qa.PIX = pixscale
    cfg = qa.INSTRUMENTS[args.instrument]
    band_upper = f"{cfg['label']} {args.tile}"

    with fits.open(scf) as h:
        sci = h[sce].data.astype(np.float64)
        shd = h[sce].header
    if 'BSOFTEN' in shd and 'BOFFSET' in shd:      # PS1 asinh → linear
        sci = (shd['BOFFSET'] + shd['BSOFTEN'] * 2.0
               * np.sinh(0.4 * np.log(10.0) * sci))
    sci = np.nan_to_num(sci.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    stars = fits.open(psf_dir / f'stars_{suffix}.fits')[2].data
    sx = np.asarray(stars['X_IMAGE'], float); sy = np.asarray(stars['Y_IMAGE'], float)
    smag = np.asarray(stars['MAG_AUTO'], float)

    # ── fit Piff on the gated stars (72_'s validated config) ──
    stamp = (args.stamp or (25 if pixscale < 0.05 else 21)) | 1   # odd window
    import piff
    samp = sci[::97, ::89].ravel(); samp = samp[np.isfinite(samp) & (samp != 0)]
    nvar = float(max(1.4826*np.median(np.abs(samp - np.median(samp))), 1e-12)**2)
    # Fit on the brightest FITMAX stars only: WISE/ground tiles carry ~1800
    # model stars and the Piff PixelGrid fit + outlier iterations scale with
    # that count (unWISE 1800 → >10 min).  The brightest few hundred fully
    # constrain a PixelGrid; more just add faint noise and cost.
    FITMAX = 400
    fsel = np.argsort(smag)[:FITMAX] if len(smag) > FITMAX else np.arange(len(smag))
    fx, fy = sx[fsel], sy[fsel]
    catf = psf_dir / f'piff_incat_{suffix}.fits'
    fits.BinTableHDU.from_columns([
        fits.Column(name='x', format='D', array=fx),
        fits.Column(name='y', format='D', array=fy)]).writeto(catf, overwrite=True)
    # Piff must fit the SAME image we score against.  For asinh-scaled data
    # (PS1) the raw file is NON-LINEAR; pointing Piff at it while giving the
    # linear noise level makes its effective noise floor ~300x too high, so
    # its Chisq outlier pass rejects all but the brightest stars — and on
    # star-poor tiles rejects EVERY star ("No stars left to fit").  Fitting a
    # PSF in asinh space is also wrong in principle.  So for asinh data we
    # hand Piff the unscaled `sci` (written to SSD, which also spares My Book
    # a 2nd full read).  Linear data (HST/JWST/unWISE) is passed through as
    # before — unchanged, so their existing panels stay valid.
    is_asinh = ('BSOFTEN' in shd and 'BOFFSET' in shd)
    imgf = None
    if is_asinh:
        imgf = Path('/tmp/piff') / f'img_{args.instrument}_{suffix}.fits'
        imgf.parent.mkdir(parents=True, exist_ok=True)
        _hdr = shd.copy()
        for _k in ('BZERO', 'BSCALE'):
            _hdr.remove(_k, ignore_missing=True)
        fits.writeto(imgf, sci, header=_hdr, overwrite=True)
        img_name, img_hdu = str(imgf), 0
    else:
        img_name, img_hdu = scf, sce
    config = {
        'input': {'image_file_name': img_name, 'image_hdu': img_hdu,
                  'cat_file_name': str(catf), 'cat_hdu': 1,
                  'x_col': 'x', 'y_col': 'y',
                  'stamp_size': stamp + 8, 'noise': nvar},
        'psf': {'model': {'type': 'PixelGrid', 'scale': pixscale, 'size': stamp},
                'interp': {'type': 'BasisPolynomial', 'order': 2},
                'outliers': {'type': 'Chisq', 'nsigma': 4.0, 'max_remove': 0.05}},
        'verbose': 0}
    try:
        piff_psf = piff.process(config)
    except Exception as e:                          # noqa: BLE001
        if imgf is not None:
            imgf.unlink(missing_ok=True)
        print(f'  PIFF FIT FAILED: {type(e).__name__}: {e}'); sys.exit(2)
    if imgf is not None:
        imgf.unlink(missing_ok=True)               # source no longer needed
    piff_psf.write(str(psf_dir / f'piff_{suffix}.piff'))
    print(f'  piff fit ok ({len(piff_psf.stars)} stars, stamp {stamp})')

    def piff_at(X, Y):        # small window for the χ² CORE comparison
        return np.asarray(piff_psf.draw(x=X, y=Y, stamp_size=stamp).array, float)

    # PSFEx at NATIVE sampling; raw_chi2 center-crops to the same ring as Piff
    # (so panel 2 compares the core, not a rescaled model — a 25px-resampled
    # PSFEx would squash the 101px model and fake a huge χ²).
    from psf_gate import psf_evaluator
    with fits.open(psf_dir / f'stars_{suffix}.psf') as _ph:
        _n = _ph[1].data['PSF_MASK'][0].shape[-1]
        native = int(round(_n * float(_ph[1].header.get('PSF_SAMP', 1.0)))) | 1
    pex_at = psf_evaluator(psf_dir / f'stars_{suffix}.psf', native)

    def piff_at_native(X, Y):  # native window → residual mosaic matches PSFEx's
        return np.asarray(piff_psf.draw(x=X, y=Y, stamp_size=native).array, float)

    # ── panel 2: mag vs chi2, both backends on the identical estimator ──
    # Cap to the brightest NMAX stars: crowded WISE/ground tiles carry ~1800
    # model stars, and the per-star χ² + neighbour loops + mosaic render blow
    # the driver's per-tile timeout.  The bright cap is the informative sample
    # (faint stars are noise-dominated) and is applied to BOTH backends.
    NMAX = 400
    sel = np.argsort(smag)[:NMAX] if len(smag) > NMAX else np.arange(len(smag))
    cx, cy, cmag = sx[sel], sy[sel], smag[sel]
    H = stamp // 2
    chi_pex = raw_chi2(sci, cx, cy, pex_at, H)
    chi_pif = raw_chi2(sci, cx, cy, piff_at, H)
    tree = cKDTree(np.column_stack([sx, sy]))          # neighbours vs ALL stars
    r1 = 1.0 / pixscale; r3 = 3.0 / pixscale
    n1 = np.array([len(tree.query_ball_point([cx[k], cy[k]], r1)) - 1 for k in range(len(cx))])
    n3 = np.array([len(tree.query_ball_point([cx[k], cy[k]], r3)) - 1 for k in range(len(cx))])
    flg0 = np.zeros(len(cx), int)
    ok = np.isfinite(chi_pex) & np.isfinite(chi_pif)
    _cap = f' (brightest {NMAX})' if len(smag) > NMAX else ''
    qa.plot_mag_vs_chi2(psf_dir / 'mag_vs_chi2_psfex.png', cmag[ok], chi_pex[ok],
                        n1[ok], n3[ok], flg0[ok], band_upper + ' — PSFEx (raw-stamp χ²)' + _cap)
    qa.plot_mag_vs_chi2(psf_dir / 'mag_vs_chi2_piff.png', cmag[ok], chi_pif[ok],
                        n1[ok], n3[ok], flg0[ok], band_upper + ' — Piff (raw-stamp χ²)' + _cap)
    print(f'  panel-2: PSFEx χ² med {np.median(chi_pex[ok]):.1f} → '
          f'Piff {np.median(chi_pif[ok]):.1f}  (n={int(ok.sum())})')

    # ── panel 5: residual mosaics vs PSFEx and vs Piff, SAME stamp window
    #    (56_ renderer, injected evaluators).  Needs the outcat (skip if the
    #    tile's outcat was cleaned away — panel 2 still stands). ──
    outf = psf_dir / f'outcat_{suffix}.fits'
    if not outf.exists():
        print('  panel-5: SKIP (no outcat)'); return
    ocd = fits.open(outf)[2].data
    mos = ocd[(np.asarray(ocd['FLAGS_PSF'], int) & 32) == 0]
    mag_mos = np.zeros(len(mos)); flg_mos = np.zeros(len(mos), int)
    for k in range(len(mos)):
        j = ((sx - mos['X_IMAGE'][k])**2 + (sy - mos['Y_IMAGE'][k])**2).argmin()
        mag_mos[k] = smag[j]
    # Piff residual mosaic at the NATIVE window so it matches the existing
    # psf_residuals.png (the PSFEx state) — a true model-only swap.
    ign = psf_dir / 'psf_samples_piff_ignore.png'
    qa.plot_raw_mosaics(ign, psf_dir / 'psf_residuals_piff.png',
                        psf_dir, sci, band_upper + ' [Piff]', suffix,
                        mos, mag_mos, flg_mos, psf_at=piff_at_native, max_show=250)
    ign.unlink(missing_ok=True)
    (psf_dir / 'psf_samples_piff_ignore_thumb.png').unlink(missing_ok=True)
    print('  panel-5: psf_residuals_piff.png (+_thumb, native window) written')


if __name__ == '__main__':
    main()
