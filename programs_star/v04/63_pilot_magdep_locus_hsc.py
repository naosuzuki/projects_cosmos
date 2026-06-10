#!/usr/bin/env python
"""
63_pilot_magdep_locus_hsc.py — PILOT: magnitude-dependent stellar-locus cut for
ground-based (HSC) PSF-star selection in the MAG vs FLUX_RADIUS (half-light)
plane.

Motivation
----------
Space data: FLUX_RADIUS of point sources is ~constant with magnitude (the PSF
is undersampled-but-stable), so a single constant [cut, upper] band works.

Ground data (HSC): FLUX_RADIUS is well-measured but its SCATTER grows toward
faint magnitudes (measurement noise ~ 1/SNR).  The builder's current band is
  med ± 3·MAD   with MAD measured GLOBALLY over Gaia stars,
i.e. set by the BRIGHT stars (MAD≈0.05 px) — far too tight for faint stars,
which scatter out of it and are wrongly rejected.

This pilot characterises the locus ridge r0(mag) and width σ(mag) on tile 40 i
and proposes a magnitude-dependent band r0(mag) ± n·σ(mag), validated against
Gaia DR3 truth where available.

Usage:  python 63_pilot_magdep_locus_hsc.py [--patch 40] [--filter i] [--explore]
"""
from __future__ import annotations
import argparse
import numpy as np
from astropy.io import fits
from pathlib import Path

WORK     = Path('/Volumes/exdisk1/data/photometry_v04')
GAIA_CSV = Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS/gaia_dr3_cosmos.csv')
PIX      = 0.168


def gaia_match(ra, dec, radius_arcsec=0.6):
    import pandas as pd
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    g = pd.read_csv(GAIA_CSV)
    rc = next(c for c in g.columns if c.lower() in ('ra', 'ra_deg', 'ra_icrs'))
    dc = next(c for c in g.columns if c.lower() in ('dec', 'dec_deg', 'dec_icrs'))
    cd = SkyCoord(ra * u.deg, dec * u.deg)
    cg = SkyCoord(g[rc].values * u.deg, g[dc].values * u.deg)
    _, d2d, _ = cd.match_to_catalog_sky(cg)
    return np.asarray(d2d.arcsec < radius_arcsec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patch', default='40')
    ap.add_argument('--filter', default='i')
    ap.add_argument('--explore', action='store_true')
    a = ap.parse_args()

    psf = WORK / f'hsc_{a.filter}' / a.patch / 'psf'
    cat = fits.open(psf / f'pass1_{a.patch}.fits')[2].data
    import json
    meta = json.loads((psf / f'psf_{a.patch}.meta.json').read_text())
    onset = float(meta['sat_onset_mag'])
    med0  = float(meta['stellar_locus_px'])
    cut0  = float(meta['artifact_cut_px'])
    upp0  = float(meta['upper_px'])

    mag = np.asarray(cat['MAG_AUTO'], float)
    fr  = np.asarray(cat['FLUX_RADIUS'], float)
    cs  = np.asarray(cat['CLASS_STAR'], float)
    snr = np.asarray(cat['SNR_WIN'], float)
    flg = np.asarray(cat['FLAGS'], int)
    ra  = np.asarray(cat['ALPHA_J2000'], float)
    dec = np.asarray(cat['DELTA_J2000'], float)

    gaia = gaia_match(ra, dec)
    finite = np.isfinite(mag) & np.isfinite(fr) & (fr > 0)
    unsat  = finite & (mag > onset)                       # below saturation
    gstar  = unsat & gaia & (flg < 2) & (snr > 8)         # Gaia truth stars
    pt     = unsat & (cs > 0.8) & (flg < 2) & (snr > 8)   # point-source proxy

    print(f'tile {a.patch} {a.filter}: onset={onset:.2f}  ridge0={med0:.3f}px  '
          f'const band=[{cut0:.3f},{upp0:.3f}]px')
    print(f'  Gaia stars (unsat,flg<2,snr>8): {gstar.sum()}   '
          f'point sources: {pt.sum()}')
    print(f'\n  mag-bin |  N_gaia med_FR  MAD  |  N_pt  med_FR  MAD  | med_SNR | '
          f'gaia in const-band')
    edges = np.arange(np.floor(onset), 26.01, 0.5)
    for lo, hi in zip(edges[:-1], edges[1:]):
        gb = gstar & (mag >= lo) & (mag < hi)
        pb = pt    & (mag >= lo) & (mag < hi)
        def stat(m):
            v = fr[m]
            if v.size == 0:
                return (0, np.nan, np.nan)
            md = np.median(v)
            return (v.size, md, 1.4826*np.median(np.abs(v-md)))
        ng, mg, dg = stat(gb)
        npt, mp, dp = stat(pb)
        snrb = np.median(snr[pb]) if pb.sum() else np.nan
        inb = (gb & (fr > cut0) & (fr < upp0)).sum()
        print(f'  {lo:4.1f}-{hi:4.1f}|  {ng:4d}  {mg:6.3f} {dg:5.3f} | '
              f'{npt:5d}  {mp:6.3f} {dp:5.3f} | {snrb:6.1f}  | '
              f'{inb:4d}/{ng:<4d} {"" if ng==0 else f"({100*inb/max(ng,1):.0f}%)"}')

    # ── tilted (linear) stellar locus from the PSFEx-ACCEPTED model stars ────
    # The accepted stars are confirmed point sources.  As a function of mag the
    # locus TILTS (faint stars lose low-SNR wings → smaller FLUX_RADIUS), so we
    # fit a STRAIGHT line FR = b·mag + a.  The artifact cut is that SAME line
    # shifted by −3·MAD, and the upper rail by +3·MAD — parallel tilted rails,
    # exactly the builder's med±3·MAD criterion but following the tilt.
    oc = fits.open(psf / f'outcat_{a.patch}.fits')
    od = next(h.data for h in oc if h.data is not None
              and getattr(h, 'columns', None) is not None and len(h.columns) > 2)
    acc = np.asarray(od['FLAGS_PSF'], int) == 0
    ox = np.asarray(od['X_IMAGE'], float)[acc]
    oy = np.asarray(od['Y_IMAGE'], float)[acc]
    cx = np.asarray(cat['X_IMAGE'], float); cy = np.asarray(cat['Y_IMAGE'], float)
    from scipy.spatial import cKDTree
    dd, jj = cKDTree(np.column_stack([cx, cy])).query(np.column_stack([ox, oy]))
    jj = jj[dd < 1.0]
    am, af = mag[jj], fr[jj]                       # accepted stars: mag, FLUX_RADIUS
    keep = np.isfinite(am) & np.isfinite(af)
    for _ in range(5):
        b1 = np.polyfit(am[keep], af[keep], 1)
        res = af - np.polyval(b1, am)
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.isfinite(am) & np.isfinite(af) & (np.abs(res) < 4.0 * mad)
    locus = np.poly1d(b1)
    cut_line = lambda mm: locus(mm) - 3.0 * mad     # artifact cut ∥ locus
    upp_line = lambda mm: locus(mm) + 3.0 * mad     # upper rail   ∥ locus
    n_acc = int(jj.size)
    print(f'\n  tilted locus from {n_acc} PSFEx-accepted stars '
          f'({keep.sum()} after clip):')
    print(f'    FR(mag) = {b1[0]:+.4f}·mag + {b1[1]:.3f} px   '
          f'(slope {b1[0]*PIX*1000:+.1f} mas/mag)   MAD = {mad:.4f} px')
    print(f'    artifact cut = locus − 3·MAD,  upper = locus + 3·MAD  (parallel)')
    for mm in (19, 21, 23, 24):
        print(f'      mag {mm}: locus={locus(mm):.3f}  '
              f'cut={cut_line(mm):.3f}  upp={upp_line(mm):.3f} px '
              f'= [{cut_line(mm)*PIX:.3f},{upp_line(mm)*PIX:.3f}]\"')

    # ── validation: Gaia-star recovery, constant vs tilted band ──
    in_const = gstar & (fr > cut0) & (fr < upp0)
    in_tilt  = gstar & (fr > cut_line(mag)) & (fr < upp_line(mag))
    print(f'\n  Gaia-star recovery: const band {in_const.sum()}/{gstar.sum()}, '
          f'tilted band {in_tilt.sum()}/{gstar.sum()}')

    if a.explore:
        return
    _make_plot(psf, a, mag, fr, cs, gaia, unsat, snr, flg, onset,
               cut0, upp0, am, af, locus, cut_line, upp_line, mad)


def _make_plot(psf, a, mag, fr, cs, gaia, unsat, snr, flg, onset,
               cut0, upp0, am, af, locus, cut_line, upp_line, mad):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 7))
    allm = unsat & (mag < 26)
    ax.scatter(mag[allm], fr[allm]*PIX, s=3, c='0.78', alpha=0.5,
               label='all detections', rasterized=True)
    ptm = allm & (cs > 0.8) & (snr > 8) & (flg < 2)
    ax.scatter(mag[ptm], fr[ptm]*PIX, s=10, c='tab:green', alpha=0.45,
               label='CLASS_STAR>0.8 point sources')
    # PSFEx-accepted model stars (the fit sample)
    ax.scatter(am, af*PIX, s=20, c='tab:blue', edgecolor='navy', linewidths=0.3,
               label=f'PSFEx-accepted stars ({am.size})', zorder=5)
    # Gaia confirmed — open red circles
    gm = allm & gaia & (flg < 2) & (snr > 8)
    ax.scatter(mag[gm], fr[gm]*PIX, s=110, facecolors='none', edgecolors='red',
               linewidths=1.5, label=f'Gaia DR3 confirmed ({gm.sum()})', zorder=7)
    # current constant band (for comparison)
    ax.axhline(cut0*PIX, color='0.4', ls=':', lw=1.3)
    ax.axhline(upp0*PIX, color='0.4', ls=':', lw=1.3,
               label=f'current CONSTANT band [{cut0*PIX:.3f},{upp0*PIX:.3f}]\"')
    # proposed tilted locus + parallel ±3·MAD rails
    mg = np.linspace(onset, 25.0, 200)
    ax.plot(mg, locus(mg)*PIX, color='k', lw=2.0, label='tilted locus (accepted stars)')
    ax.plot(mg, cut_line(mg)*PIX, color='tab:orange', lw=1.8, ls='--',
            label='artifact cut = locus − 3·MAD  (∥)')
    ax.plot(mg, upp_line(mg)*PIX, color='tab:orange', lw=1.8, ls='--',
            label='upper = locus + 3·MAD  (∥)')
    ax.fill_between(mg, cut_line(mg)*PIX, upp_line(mg)*PIX, color='tab:orange', alpha=0.13)
    ax.axvline(onset, color='magenta', ls=':', lw=1.6,
               label=f'saturation onset {onset:.2f}')
    ax.set_xlim(onset-0.4, 25.2)
    ax.set_ylim(0.35, 0.58)
    ax.set_xlabel('MAG_AUTO (HSC i 40)', fontsize=13, family='serif')
    ax.set_ylabel('FLUX_RADIUS (half-light) [arcsec]', fontsize=13, family='serif')
    ax.set_title('PILOT: tilted stellar locus + parallel ±3·MAD rails  '
                 f'(slope {(locus(25)-locus(19))*PIX/(25-19)*1000:+.1f} mas/mag)',
                 fontsize=12, family='serif')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.93)
    ax.grid(alpha=0.25)
    ax.tick_params(which='both', direction='in', top=True, right=True, length=6)
    ax.minorticks_on()
    out = psf / 'pilot_magdep_locus.png'
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches='tight'); plt.close(fig)
    print(f'\n  → {out}')


if __name__ == '__main__':
    main()
