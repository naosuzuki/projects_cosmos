#!/usr/bin/env python
"""
73_desi_match.py — extract DESI Loa spectra in the COSMOS box, match to
the master 4-way stars, and persist the result.

Inputs:
  /Users/suzuki/desiredux/zall-pix-loa.fits   (50 GB, 64M spectra)
  csvfiles_star/master_stars_4way.parquet     (9,699 master stars)

Outputs:
  csvfiles_star/desi_cosmos_box.parquet       (all 247k in-box DESI spectra)
  csvfiles_star/master_stars_4way_with_desi.parquet  (+ .csv)
       master + best DESI partner per row (tolerance 1.5") with cols
       desi_targetid, desi_survey, desi_program, desi_spectype, desi_subtype,
       desi_z, desi_zerr, desi_zwarn, desi_sep_arcsec, has_desi_spec
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u

OUT = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star')
DESI_FITS = '/Users/suzuki/desiredux/zall-pix-loa.fits'
RA_LO, RA_HI, DEC_LO, DEC_HI = 149.0, 151.1, 1.2, 3.3
MATCH_RAD_AS = 1.5


def _native(arr):
    """Force a numpy array to native byte order (FITS columns come big-endian,
    pyarrow rejects them)."""
    if hasattr(arr, 'dtype') and arr.dtype.byteorder not in ('=', '|'):
        return arr.byteswap().view(arr.dtype.newbyteorder('='))
    return arr


def main():
    t_all = time.time()

    print('[step 1] reading TARGET_RA, TARGET_DEC from DESI zall ...')
    t0 = time.time()
    with fits.open(DESI_FITS, memmap=True) as h:
        d = h['ZCATALOG'].data
        ra  = _native(np.asarray(d['TARGET_RA'], dtype=float))
        dec = _native(np.asarray(d['TARGET_DEC'], dtype=float))
    print(f'   {len(ra):,} DESI rows scanned in {time.time()-t0:.1f}s')

    sel = (ra >= RA_LO) & (ra < RA_HI) & (dec >= DEC_LO) & (dec < DEC_HI)
    idx_box = np.where(sel)[0]
    print(f'[step 2] in COSMOS box: {len(idx_box):,}')

    print('[step 3] pulling additional DESI columns for in-box rows ...')
    t0 = time.time()
    with fits.open(DESI_FITS, memmap=True) as h:
        d = h['ZCATALOG'].data
        cols = {
            'desi_targetid' : _native(np.asarray(d['TARGETID'])[idx_box]),
            'desi_survey'   : np.asarray(d['SURVEY'])[idx_box],
            'desi_program'  : np.asarray(d['PROGRAM'])[idx_box],
            'desi_spectype' : np.asarray(d['SPECTYPE'])[idx_box],
            'desi_subtype'  : np.asarray(d['SUBTYPE'])[idx_box],
            'desi_z'        : _native(np.asarray(d['Z'])[idx_box]),
            'desi_zerr'     : _native(np.asarray(d['ZERR'])[idx_box]),
            'desi_zwarn'    : _native(np.asarray(d['ZWARN'])[idx_box]),
            'desi_pmra'     : _native(np.asarray(d['PMRA'])[idx_box]),
            'desi_pmdec'    : _native(np.asarray(d['PMDEC'])[idx_box]),
            'desi_healpix'  : _native(np.asarray(d['HEALPIX'])[idx_box]),
        }
    desi = pd.DataFrame({'ra': ra[idx_box], 'dec': dec[idx_box], **cols})
    # decode bytes
    for c in ['desi_survey','desi_program','desi_spectype','desi_subtype']:
        if desi[c].dtype.kind in ('S','O'):
            desi[c] = desi[c].astype(str).str.strip()
    print(f'   pulled {len(desi):,} rows × {len(desi.columns)} cols in {time.time()-t0:.1f}s')

    desi.to_parquet(OUT/'desi_cosmos_box.parquet', index=False)
    print(f'[save] desi_cosmos_box.parquet → {len(desi):,} rows')
    print('   DESI spectype distribution in box:')
    print(desi['desi_spectype'].value_counts().to_string())

    # ── Cross-match ────────────────────────────────────────────────────────
    print('[step 4] loading master_stars_4way ...')
    m = pd.read_parquet(OUT/'master_stars_4way.parquet')
    print(f'   master: {len(m):,}')

    # best position: Gaia (propagated to 2020) if available, else JWST cat
    use_gaia = m['gaia_ra'].notna() & m['gaia_pmra'].notna()
    dt = 4.0   # rough DESI epoch (2020)
    cosd = np.cos(np.deg2rad(m['gaia_dec'].fillna(0).values))
    ra_g  = m['gaia_ra'].fillna(0).values + (m['gaia_pmra'].fillna(0).values / cosd) * dt / 3.6e6
    dec_g = m['gaia_dec'].fillna(0).values + m['gaia_pmdec'].fillna(0).values * dt / 3.6e6
    ra_m  = np.where(use_gaia,  ra_g,  m['cat_ra_jwst'].values)
    dec_m = np.where(use_gaia, dec_g,  m['cat_dec_jwst'].values)

    cm = SkyCoord(ra_m*u.deg, dec_m*u.deg)
    cd = SkyCoord(desi['ra'].values*u.deg, desi['dec'].values*u.deg)
    print(f'[step 5] cone match at {MATCH_RAD_AS}" ...')
    idx_m, idx_d, sep, _ = search_around_sky(cm, cd, MATCH_RAD_AS*u.arcsec)
    sep_as = sep.arcsec

    # best (closest) DESI per master row
    order = np.argsort(sep_as)
    seen = np.zeros(len(m), dtype=bool)
    best = np.full(len(m), -1, dtype=np.int64)
    best_sep = np.full(len(m), np.nan)
    for k in range(len(order)):
        i = order[k]; mr = idx_m[i]; dr = idx_d[i]
        if not seen[mr]:
            seen[mr] = True; best[mr] = dr; best_sep[mr] = sep_as[i]
    print(f'   unique master rows with DESI within {MATCH_RAD_AS}\": {int(seen.sum()):,}')

    # append DESI cols to master
    out = m.copy()
    for col in ['targetid','survey','program','spectype','subtype','z','zerr','zwarn','pmra','pmdec','healpix']:
        src = desi[f'desi_{col}'].values
        if src.dtype.kind in ('f','i','u'):
            arr = np.full(len(out), np.nan)
        elif src.dtype.kind == 'b':
            arr = np.full(len(out), False)
        else:
            arr = np.full(len(out), None, dtype=object)
        arr[seen] = src[best[seen]]
        out[f'desi_{col}'] = arr
    out['desi_sep_arcsec'] = best_sep
    out['has_desi_spec']   = seen

    out_pq = OUT/'master_stars_4way_with_desi.parquet'
    out_cs = OUT/'master_stars_4way_with_desi.csv'
    out.to_parquet(out_pq, index=False)
    out.to_csv(out_cs, index=False)
    print(f'\n[save] {out_pq.name}: {len(out):,} rows × {len(out.columns)} cols')

    # breakdown
    is_agn = m['is_agn_qso'].astype(bool).values
    print('\nDESI-matched master rows by source class & DESI SPECTYPE:')
    sub = out.loc[seen, ['is_agn_qso','desi_spectype']].copy()
    sub['is_agn_qso'] = sub['is_agn_qso'].astype(bool)
    crosstab = pd.crosstab(sub['is_agn_qso'].map({True:'AGN/QSO',False:'star'}),
                            sub['desi_spectype'])
    print(crosstab)
    print(f'\nWall time: {time.time()-t_all:.1f}s')


if __name__ == '__main__':
    main()
