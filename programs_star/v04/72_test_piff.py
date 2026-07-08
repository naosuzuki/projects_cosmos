#!/usr/bin/env python
"""
72_test_piff.py — Piff vs PSFEx pilot on one tile.

Fits a Piff PSF model (PixelGrid model x BasisPolynomial or GP
interpolation) on the SAME v2-gated star list the PSFEx model used
(stars_<suffix>.fits), then scores BOTH models with the identical
raw-stamp residual machinery from 56_ (amplitude + sub-pixel-shift +
background lstsq; residual as a fraction of peak) — so the comparison is
like-for-like on our data, not a brochure claim.

Outputs:
  <psf_dir>/piff_<suffix>.piff          the fitted Piff model
  <psf_dir>/piff_compare_<suffix>.png   per-star residual comparison figure
  stdout: median/95th-pct |resid|/peak + per-star chi2 for both models

Usage:
  ./72_test_piff.py --instrument hst_acs_f814w --tile A4
  ./72_test_piff.py --instrument jwst_nircam_f444w --tile A4 --suffix f444w --interp gp
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

warnings.filterwarnings('ignore')

WORK = Path('/Volumes/exdisk1/data/photometry_v04')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', required=True)
    ap.add_argument('--tile', required=True)
    ap.add_argument('--suffix', default=None,
                    help='filename suffix (JWST: the filter; default: tile id)')
    ap.add_argument('--interp', default='poly', choices=['poly', 'gp'])
    ap.add_argument('--order', type=int, default=2)
    ap.add_argument('--stamp', type=int, default=25,
                    help='Piff PixelGrid stamp size (native px)')
    return ap.parse_args()


def residual_stats(sci, xs, ys, psf_at, H):
    """56_-style raw-stamp scoring: for each star, amplitude+gradient+bg
    lstsq of the model, then |resid|/peak inside the fit ring."""
    from scipy.ndimage import shift as ndshift
    ny, nx = sci.shape
    yy, xx = np.mgrid[-H:H+1, -H:H+1]
    rad = np.hypot(xx, yy)
    fitm = rad < min(16, H - 1)
    outr = rad > min(14, 0.9 * H)
    fr_med, chi2s = [], []
    for X0, Y0 in zip(xs, ys):
        X, Y = X0 - 1.0, Y0 - 1.0          # SExtractor 1-indexed -> numpy
        xi, yi = int(round(X)), int(round(Y))
        if yi-H < 0 or xi-H < 0 or yi+H+1 > ny or xi+H+1 > nx:
            continue
        c = np.nan_to_num(sci[yi-H:yi+H+1, xi-H:xi+H+1].astype(float))
        P = psf_at(X, Y)
        if P.shape[0] != 2*H+1:            # center-crop/pad to the ring grid
            hh = P.shape[0] // 2
            if hh >= H:
                P = P[hh-H:hh+H+1, hh-H:hh+H+1]
            else:
                Q = np.zeros((2*H+1, 2*H+1)); Q[H-hh:H+hh+1, H-hh:H+hh+1] = P
                P = Q
        P = ndshift(P, (Y - yi, X - xi), order=3)
        gy, gx = np.gradient(P)
        co = np.linalg.lstsq(np.vstack([P[fitm], gx[fitm], gy[fitm],
                                        np.ones(fitm.sum())]).T,
                             c[fitm], rcond=None)[0]
        R = c - (co[0]*P + co[1]*gx + co[2]*gy + co[3])
        pk = max(float(np.nanmax(c - co[3])), 1e-9)
        sig = 1.4826*np.median(np.abs(R[outr] - np.median(R[outr]))) + 1e-9
        fr_med.append(float(np.max(np.abs(R[fitm]))) / pk)
        chi2s.append(float(np.mean((R[fitm]/sig)**2)))
    return np.asarray(fr_med), np.asarray(chi2s)


def main():
    a = parse_args()
    suffix = a.suffix or a.tile
    psf_dir = WORK / a.instrument / a.tile / 'psf'
    meta = json.loads((psf_dir / f'psf_{suffix}.meta.json').read_text())
    scf, sce = meta['sci_source_file'], int(meta.get('sci_source_ext', 0))
    pixscale = float(meta.get('pixel_scale_arcsec')
                     or meta.get('pixscale_arcsec') or 0.030)
    stars = fits.open(psf_dir / f'stars_{suffix}.fits')[2].data
    xs = np.asarray(stars['X_IMAGE'], float)
    ys = np.asarray(stars['Y_IMAGE'], float)
    print(f'{a.instrument}/{a.tile}: {len(xs)} gated model stars; '
          f'interp={a.interp} order={a.order} stamp={a.stamp}')

    with fits.open(scf) as h:
        sci = np.nan_to_num(h[sce].data.astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)

    # ── Piff fit on the SAME stars ──
    import piff, galsim, tempfile, os
    t0 = time.time()
    # measured per-pixel background variance (Piff needs a real noise level;
    # a placeholder of 1.0 makes every ~1e-3 e-/s HST pixel look like noise
    # and Piff flags ALL stars at initialization)
    samp = sci[::97, ::89].ravel()
    samp = samp[np.isfinite(samp) & (samp != 0)]
    med = np.median(samp)
    sig = 1.4826 * np.median(np.abs(samp - med))
    noise_var = float(max(sig, 1e-12)**2)
    print(f'  background sigma = {sig:.4g}  -> noise var {noise_var:.4g}')
    # write a minimal star catalog for Piff's input module
    catf = psf_dir / f'piff_incat_{suffix}.fits'
    fits.BinTableHDU.from_columns([
        fits.Column(name='x', format='D', array=xs),
        fits.Column(name='y', format='D', array=ys),
    ]).writeto(catf, overwrite=True)
    # Piff wants an image file; give it the original (ext handled via hdu)
    interp_cfg = ({'type': 'BasisPolynomial', 'order': a.order}
                  if a.interp == 'poly' else
                  {'type': 'GPInterp', 'kernel': f'RBF(300*(0.26/0.03))',
                   'optimizer': 'none'})
    config = {
        'input': {
            'image_file_name': scf, 'image_hdu': sce,
            'cat_file_name': str(catf), 'cat_hdu': 1,
            'x_col': 'x', 'y_col': 'y',
            # input stamps must EXCEED the model grid footprint: the
            # PixelGrid is defined in sky arcsec and the WCS's true
            # pixel scale is fractionally under the nominal one
            'stamp_size': a.stamp + 8,
            'noise': noise_var,
        },
        'psf': {
            # Piff models in SKY coords: scale is arcsec/px of the model
            'model': {'type': 'PixelGrid', 'scale': pixscale, 'size': a.stamp},
            'interp': interp_cfg,
            'outliers': {'type': 'Chisq', 'nsigma': 4.0, 'max_remove': 0.05},
        },
        'verbose': 1,
    }
    psf_piff = piff.process(config)
    print(f'  piff fit: {time.time()-t0:.0f}s  '
          f'(stars used: {len(psf_piff.stars)})')
    psf_piff.write(str(psf_dir / f'piff_{suffix}.piff'))

    def piff_at(X, Y):
        img = psf_piff.draw(x=X, y=Y, stamp_size=a.stamp)
        return np.asarray(img.array, float)

    # ── PSFEx evaluator at its NATIVE size (residual_stats center-crops
    #    to the scoring window; forcing psf_size=stamp would squash the
    #    whole 101-px model footprint into the stamp) ──
    from psf_gate import psf_evaluator
    with fits.open(psf_dir / f'stars_{suffix}.psf') as _ph:
        _hd = _ph[1].header
        _mk_n = _ph[1].data['PSF_MASK'][0].shape[-1]
        native = int(round(_mk_n * float(_hd.get('PSF_SAMP', 1.0)))) | 1
    pex_at = psf_evaluator(psf_dir / f'stars_{suffix}.psf', native)

    H = a.stamp // 2
    fr_pex, chi_pex = residual_stats(sci, xs, ys, pex_at, H)
    fr_pif, chi_pif = residual_stats(sci, xs, ys, piff_at, H)

    def s(v):
        return f'median {np.median(v):.4f}  p95 {np.percentile(v, 95):.4f}'
    print(f'  PSFEx : max|resid|/peak  {s(fr_pex)}   chi2 {s(chi_pex)}')
    print(f'  Piff  : max|resid|/peak  {s(fr_pif)}   chi2 {s(chi_pif)}')

    # comparison figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n = min(len(fr_pex), len(fr_pif))
    axes[0].scatter(fr_pex[:n]*100, fr_pif[:n]*100, s=14, alpha=0.7)
    lim = max(np.percentile(fr_pex[:n], 99), np.percentile(fr_pif[:n], 99))*100
    axes[0].plot([0, lim], [0, lim], 'k:', lw=1)
    axes[0].set_xlabel('PSFEx  max|resid|/peak  (%)')
    axes[0].set_ylabel('Piff  max|resid|/peak  (%)')
    axes[0].set_title(f'{a.instrument} {a.tile} — per-star residuals '
                      f'(below line = Piff better)')
    axes[1].hist([chi_pex[:n], chi_pif[:n]],
                 bins=np.geomspace(max(min(chi_pex.min(), chi_pif.min()), .1),
                                   max(chi_pex.max(), chi_pif.max()), 30),
                 label=['PSFEx', 'Piff'], histtype='step', lw=2)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('per-star residual χ²'); axes[1].legend()
    fig.tight_layout()
    out_png = psf_dir / f'piff_compare_{suffix}.png'
    fig.savefig(out_png, dpi=110)
    print(f'  wrote {out_png}')


if __name__ == '__main__':
    main()
