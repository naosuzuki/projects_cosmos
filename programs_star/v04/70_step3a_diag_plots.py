#!/usr/bin/env python
"""
70_step3a_diag_plots.py — Step 3a visual-inspection diagnostics (per
mission, tile).  STARS ONLY (3b is_star).  Requested 2026-06-11:

  Plot 1  diag1_stars_radec_<tile>.png
      RA/Dec of stars inside the tile outline; color = magnitude
      (red bright → blue faint), point size also tracks brightness;
      SOLID circles for isolated stars (n₁=0), OPEN circles for stars
      with ≥1 neighbour within 1″ (n₁>0; neighbours counted in the
      full merged detection catalog).
  Plot 2  diag2_phot_compare_<tile>.png
      per band, three columns:
        APER−PSFEx  vs PSFEx mag
        APER−DAO    vs DAO mag
        PSFEx−DAO   vs PSFEx mag
      (aperture = SExtractor MAG_APER #2 ≈ 0.5″ diameter, from the
      per-band dual catalogs).  Same open/solid n₁ convention.
  Plot 3  diag3_psf_composite_<tile>.png
      drizzle-style shift-and-add composite of all clean star stamps
      (unsaturated, n₁=0), oversampled ×9, simply summed; per band,
      sqrt stretch, stamp radius ±1.2″.
  Plot 4  diag4_psf_profile_<tile>.png
      horizontal profile cut through the composite peak, x in arcsec
      (native px on the top axis), y normalized to the peak.

Usage:  ./70_step3a_diag_plots.py --mission jwst --tile A4
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, join
from astropy.wcs import WCS
from scipy.spatial import cKDTree

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

OVER = 9                      # plot-3 oversampling factor
PSF_RADIUS_ARCSEC = 1.2       # plot-3/4 composite stamp radius


def clean_mag(a) -> np.ndarray:
    """float array with SExtractor ±99 sentinels → NaN."""
    m = np.asarray(a, dtype=float).copy()
    m[(m <= -90) | (m >= 90)] = np.nan
    return m


def tile_outline(chi2_path: Path):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h = fits.open(chi2_path, memmap=True)[0]
        w = WCS(h.header)
        ny, nx = h.data.shape
        c = w.pixel_to_world([0, nx-1, nx-1, 0, 0], [0, 0, ny-1, ny-1, 0])
    return c.ra.deg, c.dec.deg


def n1_neighbours(stars: Table, merged: Table) -> np.ndarray:
    """# of OTHER merged-catalog sources within 1 arcsec of each star."""
    dec0 = np.median(np.asarray(merged['DELTA_J2000'], dtype=float))
    cosd = np.cos(np.deg2rad(dec0))
    xy_all = np.column_stack(
        [np.asarray(merged['ALPHA_J2000'], dtype=float) * cosd,
         np.asarray(merged['DELTA_J2000'], dtype=float)])
    xy_st = np.column_stack(
        [np.asarray(stars['ALPHA_J2000'], dtype=float) * cosd,
         np.asarray(stars['DELTA_J2000'], dtype=float)])
    tree = cKDTree(xy_all)
    counts = tree.query_ball_point(xy_st, r=1.0 / 3600.0,
                                   return_length=True)
    return np.maximum(counts - 1, 0)        # exclude self


def split_markers(ax, x, y, n1, **kw):
    """solid circles for n1==0, open circles for n1>0."""
    iso = n1 == 0
    c = kw.pop('c', None)
    s = kw.pop('s', 25)
    cmap = kw.pop('cmap', None)
    vmin = kw.pop('vmin', None)
    vmax = kw.pop('vmax', None)
    sc = None
    if np.any(iso):
        sc = ax.scatter(np.asarray(x)[iso], np.asarray(y)[iso],
                        c=(np.asarray(c)[iso] if c is not None else None),
                        s=(np.asarray(s)[iso] if np.ndim(s) else s),
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        edgecolors='none', **kw)
    if np.any(~iso):
        if c is not None and cmap is not None:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            edge = plt.get_cmap(cmap)(norm(np.asarray(c)[~iso]))
        else:
            edge = 'k'
        ax.scatter(np.asarray(x)[~iso], np.asarray(y)[~iso],
                   facecolors='none', edgecolors=edge,
                   s=(np.asarray(s)[~iso] if np.ndim(s) else s),
                   linewidths=1.1, **kw)
    return sc


def load_aperture_mags(phot_dir: Path, band: str, tile: str) -> Table:
    """MAG_APER #2 from the per-band dual catalogs, keyed by
    (DETECT_MODE, NUMBER)."""
    parts = []
    for mode in ('cold', 'hot'):
        p = phot_dir / f'cat_{band}_{mode}_psf.fits'
        if not p.exists():
            continue
        d = fits.open(p)[2].data
        t = Table()
        t['NUMBER'] = d['NUMBER']
        t['DETECT_MODE'] = np.full(len(t), mode)
        ap = np.atleast_2d(d['MAG_APER'])
        t[f'{band}_MAG_APER'] = ap[:, 1] if ap.shape[1] > 1 else ap[:, 0]
        parts.append(t)
    from astropy.table import vstack
    return vstack(parts) if parts else Table()


# ----------------------------------------------------------------------
def plot1_radec(stars, n1, mag, outline, out, tile, band_ref):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ok = np.isfinite(mag)
    vmin, vmax = (np.percentile(mag[ok], [2, 98]) if ok.sum() > 3
                  else (16, 28))
    size = 12 + 140 * np.clip((vmax - mag) / max(vmax - vmin, 1e-3), 0, 1)
    sc = split_markers(ax, np.asarray(stars['ALPHA_J2000'], dtype=float),
                       np.asarray(stars['DELTA_J2000'], dtype=float),
                       n1, c=mag, s=size, cmap='RdYlBu',
                       vmin=vmin, vmax=vmax)
    ax.plot(outline[0], outline[1], '-', color='0.3', lw=1.2, zorder=0)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label(f'{band_ref} MAG_PSF  (red = bright)')
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Dec [deg]')
    ax.invert_xaxis()
    ax.set_aspect('equal', adjustable='datalim')
    n_iso = int((n1 == 0).sum())
    ax.set_title(f'{tile}: {len(stars)} stars  '
                 f'(solid: isolated {n_iso}, open: n₁>0 {len(stars)-n_iso})')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot2_compare(stars, n1, bands, out, tile):
    rows = [b for b in bands if f'{b}_MAG_PSF' in stars.colnames]
    fig, axes = plt.subplots(len(rows), 3,
                             figsize=(13.5, 3.4 * len(rows)),
                             squeeze=False)
    for i, b in enumerate(rows):
        psf = clean_mag(stars[f'{b}_MAG_PSF'])
        ap = (clean_mag(stars[f'{b}_MAG_APER'])
              if f'{b}_MAG_APER' in stars.colnames
              else np.full(len(stars), np.nan))
        dao = (clean_mag(stars[f'{b}_DAO_MAG'])
               if f'{b}_DAO_MAG' in stars.colnames
               else np.full(len(stars), np.nan))
        panels = [(psf, ap - psf, 'PSFEx mag', 'APER − PSFEx'),
                  (dao, ap - dao, 'DAO mag', 'APER − DAO'),
                  (psf, psf - dao, 'PSFEx mag', 'PSFEx − DAO')]
        for k, (x, y, xl, yl) in enumerate(panels):
            ax = axes[i][k]
            f = np.isfinite(x) & np.isfinite(y)
            split_markers(ax, x[f], y[f], n1[f], c=None, s=22)
            if f.sum():
                med = np.median(y[f])
                ax.axhline(med, color='tab:red', lw=0.8, ls='--',
                           label=f'median {med:+.3f}')
                ax.legend(fontsize=8, loc='upper left')
            ax.axhline(0, color='0.6', lw=0.6)
            ax.set_xlabel(f'{b} {xl}')
            ax.set_ylabel(yl)
            lim = (np.percentile(np.abs(y[f] - np.median(y[f])), 98) * 3
                   if f.sum() > 3 else 1)
            ax.set_ylim(np.median(y[f]) - max(lim, 0.3),
                        np.median(y[f]) + max(lim, 0.3))
    fig.suptitle(f'{tile} stars: aperture vs PSF-fit photometry '
                 f'(solid n₁=0, open n₁>0)', y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def drizzle_composite(sci, xs, ys, half_native: int):
    """Shift-and-add star stamps on a ×OVER grid — drizzle with
    pixfrac=1: each native pixel's flux is spread over its full
    OVER×OVER fine-pixel footprint (np.repeat), then the whole fine
    stamp is integer-shifted by the star's fractional-centroid offset
    in fine pixels and summed.  (A first version deposited each native
    pixel on a SINGLE subpixel, leaving an 8/9-empty comb pattern.)"""
    size = (2 * half_native + 1) * OVER
    comp = np.zeros((size + 2 * OVER, size + 2 * OVER), dtype=np.float64)
    nused = 0
    ny, nx = sci.shape
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if not (half_native + 1 <= xi < nx - half_native - 1 and
                half_native + 1 <= yi < ny - half_native - 1):
            continue
        stamp = sci[yi-half_native:yi+half_native+1,
                    xi-half_native:xi+half_native+1].astype(np.float64)
        if not np.isfinite(stamp).all():
            continue
        rim = np.concatenate([stamp[0], stamp[-1], stamp[:, 0],
                              stamp[:, -1]])
        stamp = stamp - np.median(rim)
        fine = np.repeat(np.repeat(stamp, OVER, axis=0), OVER, axis=1)
        # shift so the centroid lands on the composite center
        sx = int(round((xi - x) * OVER))
        sy = int(round((yi - y) * OVER))
        y0 = OVER + sy
        x0 = OVER + sx
        comp[y0:y0+size, x0:x0+size] += fine
        nused += 1
    return comp[OVER:-OVER, OVER:-OVER], nused


def plots34_psf(stars, n1, bands, mission, tile, spec, phot_dir):
    fig4, ax4 = plt.subplots(figsize=(8.5, 5.5))
    pixscale = spec['pixscale']
    info = {}
    for k, b in enumerate(bands):
        if f'{b}_MAG_PSF' not in stars.colnames:
            continue
        figb, ax = plt.subplots(figsize=(6.5, 6.2))
        sat = np.asarray(stars[f'{b}_IS_SATURATED'], dtype=bool)
        mag = clean_mag(stars[f'{b}_MAG_PSF'])
        use = (~sat) & (n1 == 0) & np.isfinite(mag)
        _, _, loader = band_measurement(mission, tile, b)
        if loader is None or use.sum() == 0:
            plt.close(figb)
            continue
        sci = np.asarray(loader(), dtype=np.float32)
        half = int(round(PSF_RADIUS_ARCSEC / pixscale))   # fixed ±1.2"
        comp, nused = drizzle_composite(
            sci, np.asarray(stars['X_IMAGE'], dtype=float)[use] - 1,
            np.asarray(stars['Y_IMAGE'], dtype=float)[use] - 1, half)
        del sci
        if comp.max() <= 0:
            plt.close(figb)
            continue
        comp /= comp.max()
        ext = (np.array([-1, 1, -1, 1]) * (half + 0.5) * pixscale)
        ax.imshow(np.sqrt(np.clip(comp, 0, 1)), origin='lower',
                  cmap='inferno', vmin=0, vmax=1, extent=ext)
        ax.set_title(f'{tile} {b} composite PSF — drizzle ×{OVER}, '
                     f'{nused} clean stars, sqrt stretch', fontsize=10)
        ax.set_xlabel('arcsec')
        ax.set_ylabel('arcsec')
        figb.tight_layout()
        out3b = phot_dir / f'diag3_psf_composite_{tile}_{b}.png'
        figb.savefig(out3b, dpi=130)
        plt.close(figb)
        # plot 4 per band + the cross-band overlay
        py, px = np.unravel_index(np.argmax(comp), comp.shape)
        cut = comp[py]
        xax = (np.arange(comp.shape[1]) - px) / OVER * pixscale
        fig4b, ax4b = plt.subplots(figsize=(8.5, 5.5))
        ax4b.plot(xax, cut, lw=1.4, color='tab:red')
        ax4b.set_xlabel('offset [arcsec]')
        sec = ax4b.secondary_xaxis('top', functions=(
            lambda a: a / pixscale, lambda q: q * pixscale))
        sec.set_xlabel('offset [native pixel]')
        ax4b.set_ylabel('normalized at peak')
        ax4b.set_yscale('log')
        ax4b.set_ylim(1e-5, 1.5)
        ax4b.axhline(0.5, color='0.7', lw=0.6, ls=':')
        ax4b.set_title(f'{tile} {b} composite-PSF profile cut '
                       f'(×{OVER} drizzle, {nused}★)')
        fig4b.tight_layout()
        fig4b.savefig(phot_dir / f'diag4_psf_profile_{tile}_{b}.png',
                      dpi=130)
        plt.close(fig4b)
        ax4.plot(xax, cut, lw=1.4, label=f'{b} ({nused}★)')
        info[b] = nused
    ax4.set_xlabel('offset [arcsec]')
    secax = ax4.secondary_xaxis(
        'top', functions=(lambda a: a / pixscale, lambda p: p * pixscale))
    secax.set_xlabel('offset [native pixel]')
    ax4.set_ylabel('normalized at peak')
    ax4.set_yscale('log')
    ax4.set_ylim(1e-5, 1.5)
    ax4.axhline(0.5, color='0.7', lw=0.6, ls=':')
    ax4.legend()
    ax4.set_title(f'{tile} composite-PSF profile cuts (×{OVER} drizzle)')
    fig4.tight_layout()
    fig4.savefig(phot_dir / f'diag4_psf_profile_{tile}.png', dpi=130)
    plt.close(fig4)
    return info


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mission', required=True, choices=sorted(MISSIONS))
    ap.add_argument('--tile', required=True)
    ap.add_argument('--band-ref', default=None,
                    help='band for plot-1 colors (default: 2nd band)')
    args = ap.parse_args()
    spec = MISSIONS[args.mission]
    tdir = WORK / f'{args.mission}_chi2' / args.tile
    phot_dir = tdir / 'phot'
    t0 = time.time()

    star_p = phot_dir / f'star_{args.tile}.fits'
    phot_p = phot_dir / f'phot_{args.tile}.fits'
    if not star_p.exists() or not phot_p.exists():
        sys.exit(f'need 53_+67_ outputs in {phot_dir}')
    star = Table.read(star_p)
    star = star[np.asarray(star['is_star'], dtype=bool)]
    phot = Table.read(phot_p)
    merged = phot                      # spine == merged catalog columns
    bands = [b for b in spec['bands'] if f'{b}_MAG_PSF' in phot.colnames]

    # star rows with full photometry columns
    stars = join(star['SOURCE_ID',], phot, keys='SOURCE_ID',
                 join_type='left')
    # aperture mags from the raw dual catalogs
    for b in bands:
        apt = load_aperture_mags(phot_dir, b, args.tile)
        if len(apt):
            stars = join(stars, apt, keys=['DETECT_MODE', 'NUMBER'],
                         join_type='left')
    # DAO mags
    dao_p = phot_dir / f'dao_{args.tile}.fits'
    if dao_p.exists():
        dao = Table.read(dao_p)
        keep = ['SOURCE_ID'] + [c for c in dao.colnames
                                if c.endswith('_DAO_MAG')]
        stars = join(stars, dao[keep], keys='SOURCE_ID', join_type='left')

    n1 = n1_neighbours(stars, merged)
    outline = tile_outline(tdir / f'chi2_{args.tile}.fits')
    print(f'{args.mission}/{args.tile}: {len(stars)} stars, '
          f'{int((n1 == 0).sum())} isolated; bands {bands}')

    for b in bands:
        mag_b = clean_mag(stars[f'{b}_MAG_PSF'])
        p1 = phot_dir / f'diag1_stars_radec_{args.tile}_{b}.png'
        plot1_radec(stars, n1, mag_b, outline, p1, args.tile, b)
        p2 = phot_dir / f'diag2_phot_compare_{args.tile}_{b}.png'
        plot2_compare(stars, n1, [b], p2, args.tile)
    print(f'  [save] diag1/diag2 per band ({len(bands)} each)')
    used = plots34_psf(stars, n1, bands, args.mission, args.tile, spec,
                       phot_dir)
    print(f'  [save] diag3/diag4 per band + overlay (stamps: {used})')
    print(f'=== diag {args.mission}/{args.tile} done '
          f'({time.time()-t0:.0f}s) ===')


if __name__ == '__main__':
    main()
