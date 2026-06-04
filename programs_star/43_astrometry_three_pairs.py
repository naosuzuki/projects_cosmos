#!/usr/bin/env python
"""
43_astrometry_three_pairs.py — astrometric residuals for HST-Euclid,
                                JWST-Euclid, HST-JWST × stars / galaxies.

Following projects_euclid/programs/19_astrometry_three_pairs.py.

Outputs:
  csvfiles_star/astrometry_three_pairs.csv
  csvfiles_star/astrometry_three_pairs_summary.txt
  htmls/star_v01/astrometry_three_pairs_2d.png       (3 pairs × stars/gals)
  htmls/star_v01/astrometry_three_pairs_vs_mag.png   (residual vs mag)
"""
from __future__ import annotations
import warnings
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy import units as u

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'star_v01'

MATCH_AS = 1.0
SHARP_STAR_RANGE = (0.50, 0.75)
SHARP_GAL_RANGE  = (0.20, 0.40)
SNR_MIN = 5.0


def nmad(x):
    x = np.asarray(x)
    if len(x) == 0:
        return np.nan
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))


def mutual_nn(c_a, c_b, rad):
    if len(c_a) == 0 or len(c_b) == 0:
        return np.zeros(len(c_a), dtype=bool), np.zeros(len(c_a), dtype=int)
    a2b_idx, a2b_sep, _ = c_a.match_to_catalog_sky(c_b)
    b2a_idx, b2a_sep, _ = c_b.match_to_catalog_sky(c_a)
    ok_a = a2b_sep <= rad
    ok_b = b2a_sep <= rad
    mutual = ok_a & ok_b[a2b_idx] & (b2a_idx[a2b_idx] == np.arange(len(c_a)))
    return mutual, a2b_idx


def load_common_mask():
    p = OUT / 'footprints' / 'common_footprint_v5.fits'
    with fits.open(p) as h:
        return h[0].data.astype(bool), WCS(h[0].header)


def inside_mask(ra, dec, mask, wcs):
    x, y = wcs.all_world2pix(ra, dec, 0)
    xi = np.round(x).astype(int); yi = np.round(y).astype(int)
    ny, nx = mask.shape
    ok = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    out = np.zeros(len(ra), dtype=bool)
    out[ok] = mask[yi[ok], xi[ok]]
    return out


def load_band(band, common, common_wcs):
    dao = pd.read_parquet(OUT / 'dao' / f'dao_{band}.parquet')
    dao['snr'] = dao['peak'] / dao['sky_std']
    in_com = inside_mask(dao['ra'].values, dao['dec'].values, common, common_wcs)
    df = dao[in_com].reset_index(drop=True)
    is_star = ((df['sharpness'].between(*SHARP_STAR_RANGE)) &
               (df['roundness1'].abs() <= 0.5) & (df['roundness2'].abs() <= 0.5) &
               (df['snr'] >= SNR_MIN))
    is_gal  = ((df['sharpness'].between(*SHARP_GAL_RANGE)) &
               (df['roundness1'].abs() <= 0.5) & (df['roundness2'].abs() <= 0.5) &
               (df['snr'] >= SNR_MIN))
    return df[is_star].reset_index(drop=True), df[is_gal].reset_index(drop=True)


def residuals_mas(ra_a, dec_a, ra_b, dec_b):
    cos_d = np.cos(np.deg2rad(dec_b))
    dra  = (ra_a  - ra_b)  * cos_d * 3.6e6
    ddec = (dec_a - dec_b)         * 3.6e6
    return np.asarray(dra), np.asarray(ddec)


def per_mag_stats(mag, dra, ddec, mag_bins):
    rows = []
    for i in range(len(mag_bins) - 1):
        m = (mag >= mag_bins[i]) & (mag < mag_bins[i+1])
        if int(m.sum()) < 5:
            continue
        rows.append(dict(
            mag_lo=mag_bins[i], mag_hi=mag_bins[i+1], n=int(m.sum()),
            med_dra=float(np.median(dra[m])),  med_ddec=float(np.median(ddec[m])),
            nmad_dra=float(nmad(dra[m])),       nmad_ddec=float(nmad(ddec[m])),
        ))
    return rows


def main():
    common, common_wcs = load_common_mask()
    print(f'Common region: {int(common.sum()):,} px = '
          f'{common.sum()*(3.0/60)**2:.1f} arcmin²')

    print('\nLoading DAO catalogs restricted to common region:')
    h_star, h_gal = load_band('F814W', common, common_wcs)
    j_star, j_gal = load_band('F115W', common, common_wcs)
    e_star, e_gal = load_band('VIS',   common, common_wcs)
    print(f'  HST    : stars {len(h_star):,}  gals {len(h_gal):,}')
    print(f'  JWST   : stars {len(j_star):,}  gals {len(j_gal):,}')
    print(f'  Euclid : stars {len(e_star):,}  gals {len(e_gal):,}')

    coords = {'HST':    (h_star, h_gal),
              'JWST':   (j_star, j_gal),
              'Euclid': (e_star, e_gal)}

    rad = MATCH_AS * u.arcsec
    pairs = [('HST', 'Euclid'), ('JWST', 'Euclid'), ('HST', 'JWST')]
    results = {}
    mag_bins = np.arange(17.0, 28.5 + 0.01, 0.5)

    for A, B in pairs:
        print(f'\n=== {A} - {B} ===')
        Astar, Agal = coords[A]
        Bstar, Bgal = coords[B]

        # STARS
        cA = SkyCoord(Astar['ra'].values * u.deg, Astar['dec'].values * u.deg)
        cB = SkyCoord(Bstar['ra'].values * u.deg, Bstar['dec'].values * u.deg)
        mut, idx = mutual_nn(cA, cB, rad)
        ra_a = Astar['ra'].values[mut];     dec_a = Astar['dec'].values[mut]
        ra_b = Bstar['ra'].values[idx[mut]]; dec_b = Bstar['dec'].values[idx[mut]]
        dra_s, ddec_s = residuals_mas(ra_a, dec_a, ra_b, dec_b)
        mag_s = Astar['mag'].values[mut]

        # GALAXIES
        cA = SkyCoord(Agal['ra'].values * u.deg, Agal['dec'].values * u.deg)
        cB = SkyCoord(Bgal['ra'].values * u.deg, Bgal['dec'].values * u.deg)
        mut, idx = mutual_nn(cA, cB, rad)
        ra_a = Agal['ra'].values[mut];     dec_a = Agal['dec'].values[mut]
        ra_b = Bgal['ra'].values[idx[mut]]; dec_b = Bgal['dec'].values[idx[mut]]
        dra_g, ddec_g = residuals_mas(ra_a, dec_a, ra_b, dec_b)
        mag_g = Agal['mag'].values[mut]

        def summary(dra, ddec):
            return dict(N=len(dra),
                        med_dra=float(np.median(dra)) if len(dra) else np.nan,
                        med_ddec=float(np.median(ddec)) if len(ddec) else np.nan,
                        nmad_dra=float(nmad(dra)),
                        nmad_ddec=float(nmad(ddec)))

        sum_s = summary(dra_s, ddec_s)
        sum_g = summary(dra_g, ddec_g)
        print(f'  STARS    N={sum_s["N"]:>6,d}  '
              f'med ({sum_s["med_dra"]:+.2f}, {sum_s["med_ddec"]:+.2f})  '
              f'NMAD ({sum_s["nmad_dra"]:.2f}, {sum_s["nmad_ddec"]:.2f})')
        print(f'  GALAXIES N={sum_g["N"]:>6,d}  '
              f'med ({sum_g["med_dra"]:+.2f}, {sum_g["med_ddec"]:+.2f})  '
              f'NMAD ({sum_g["nmad_dra"]:.2f}, {sum_g["nmad_ddec"]:.2f})')

        results[(A, B)] = dict(
            star=dict(summary=sum_s, dra=dra_s, ddec=ddec_s, mag=mag_s),
            gal =dict(summary=sum_g, dra=dra_g, ddec=ddec_g, mag=mag_g),
        )

    # Text summary
    out_txt = OUT / 'astrometry_three_pairs_summary.txt'
    with open(out_txt, 'w') as fh:
        fh.write('Astrometric residuals — three survey pairs × stars/galaxies\n')
        fh.write(f'Common region: HST F814W ∩ JWST F115W pixel coverage\n')
        fh.write(f'Match radius: mutual-NN ≤ {MATCH_AS} arcsec\n')
        fh.write(f'Star criterion : DAO sharp ∈ {SHARP_STAR_RANGE}, |round|≤0.5, SNR≥5\n')
        fh.write(f'Galaxy criterion: DAO sharp ∈ {SHARP_GAL_RANGE}, |round|≤0.5, SNR≥5\n')
        fh.write('Residual = A - B (mas), with Δα · cosδ on the RA axis\n\n')
        for A, B in pairs:
            r = results[(A, B)]
            fh.write(f'=== {A} - {B} ===\n')
            for pop in ('star', 'gal'):
                s = r[pop]['summary']
                fh.write(f'  {pop.upper():>8}  N={s["N"]:>7,d}  '
                         f'med (dRA*, dDec) = ({s["med_dra"]:+7.2f}, {s["med_ddec"]:+7.2f}) mas  '
                         f'NMAD ({s["nmad_dra"]:.2f}, {s["nmad_ddec"]:.2f})\n')
            fh.write('\n')
    print(f'\nWrote {out_txt}')

    out_csv = OUT / 'astrometry_three_pairs.csv'
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['pair','population','N','med_dRA_mas','med_dDec_mas',
                    'NMAD_dRA_mas','NMAD_dDec_mas'])
        for A, B in pairs:
            for pop in ('star', 'gal'):
                s = results[(A, B)][pop]['summary']
                w.writerow([f'{A}-{B}', pop, s['N'],
                            f'{s["med_dra"]:+.2f}', f'{s["med_ddec"]:+.2f}',
                            f'{s["nmad_dra"]:.2f}', f'{s["nmad_ddec"]:.2f}'])
    print(f'Wrote {out_csv}')

    # 2D residual plots
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, (A, B) in enumerate(pairs):
        for row, pop in enumerate(('star', 'gal')):
            ax = axes[row, col]
            r = results[(A, B)][pop]
            d_ra, d_dec = r['dra'], r['ddec']
            if len(d_ra) == 0:
                ax.text(0.5, 0.5, 'no sources', ha='center', va='center',
                        transform=ax.transAxes); continue
            ax.axhline(0, color='0.6', lw=0.6, ls='--')
            ax.axvline(0, color='0.6', lw=0.6, ls='--')
            ax.plot(d_ra, d_dec, '.', ms=1.0, alpha=0.10,
                    color='C0' if pop == 'star' else 'C3', rasterized=True)
            med_dra, med_ddec = np.median(d_ra), np.median(d_dec)
            ax.plot(med_dra, med_ddec, '+', ms=14, mew=2.0, color='k',
                    label=f'median ({med_dra:+.1f}, {med_ddec:+.1f})')
            ax.set_xlim(-200, 200); ax.set_ylim(-200, 200)
            ax.set_aspect('equal'); ax.grid(alpha=0.3)
            ax.set_title(f'{A} - {B}  ({pop}, N={len(d_ra):,})')
            ax.set_xlabel(r'$\Delta\alpha\cos\delta$ (mas)')
            if col == 0:
                ax.set_ylabel(r'$\Delta\delta$ (mas)')
            ax.legend(loc='upper left', fontsize=8)
    fig.suptitle('Astrometric residuals (mas) — three survey pairs × stars/galaxies\n'
                 'restricted to HST F814W ∩ JWST F115W common pixel coverage')
    fig.tight_layout()
    p = HTML / 'astrometry_three_pairs_2d.png'
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f'Wrote {p}')

    # Residual vs magnitude
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for col, (A, B) in enumerate(pairs):
        for row, pop in enumerate(('star', 'gal')):
            ax = axes[row, col]
            r = results[(A, B)][pop]
            mag = r['mag']
            d_ra, d_dec = r['dra'], r['ddec']
            if len(mag) == 0:
                continue
            ax.axhline(0, color='0.6', lw=0.6, ls='--')
            ax.plot(mag, d_ra,  '.', ms=1, alpha=0.06, color='C0',
                    rasterized=True, label='dRA*')
            ax.plot(mag, d_dec, '.', ms=1, alpha=0.06, color='C3',
                    rasterized=True, label='dDec')
            rows = per_mag_stats(mag, d_ra, d_dec, mag_bins)
            if rows:
                centres = [0.5*(rr['mag_lo']+rr['mag_hi']) for rr in rows]
                med_dra = [rr['med_dra'] for rr in rows]
                med_dde = [rr['med_ddec'] for rr in rows]
                ax.plot(centres, med_dra, 'o-', color='C0', lw=1.5, label='med dRA*')
                ax.plot(centres, med_dde, 's-', color='C3', lw=1.5, label='med dDec')
            ax.set_ylim(-150, 150); ax.grid(alpha=0.3)
            ax.set_title(f'{A} - {B}  ({pop}, N={len(mag):,})')
            if row == 1: ax.set_xlabel('DAO mag (left survey)')
            if col == 0: ax.set_ylabel('residual (mas)')
            ax.legend(loc='upper left', fontsize=7)
    fig.suptitle('Residual vs DAO instrumental magnitude — three survey pairs')
    fig.tight_layout()
    p2 = HTML / 'astrometry_three_pairs_vs_mag.png'
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f'Wrote {p2}')

    print('\n' + open(out_txt).read())


if __name__ == '__main__':
    main()
