#!/usr/bin/env python
"""
70_ps1_sdss_pull.py — fetch Pan-STARRS DR1 and SDSS catalogs over the COSMOS
field box (149.0 ≤ RA ≤ 151.1, 1.2 ≤ Dec ≤ 3.3 deg).

Pan-STARRS:  via VizieR II/349 (PS1 mean PSF magnitudes, ugrizy).
SDSS:        via astroquery.sdss DR16 photometric (ugriz PSF mags +
             classification type).  Tiled in a 5×5 grid because the
             SQL endpoint caps each query at ~3k rows.

Outputs:
  csvfiles_star/ps1_cosmos.parquet  (+ .csv)
  csvfiles_star/sdss_cosmos.parquet (+ .csv)
"""
from __future__ import annotations
import time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'

RA_LO, RA_HI   = 149.0, 151.1
DEC_LO, DEC_HI = 1.2, 3.3


def pull_ps1():
    print('[PS1] fetching from VizieR II/349 ...')
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    Vizier.ROW_LIMIT = -1
    cols = ['objID', 'RAJ2000', 'DEJ2000',
            'gmag', 'e_gmag', 'rmag', 'e_rmag',
            'imag', 'e_imag', 'zmag', 'e_zmag', 'ymag', 'e_ymag',
            'Ng', 'Nr', 'Ni', 'Nz', 'Ny']
    v = Vizier(columns=cols, row_limit=-1)
    # one big cone covering the box (center 150.05, 2.25; diag ≈ 1.48°)
    pos = SkyCoord(0.5*(RA_LO+RA_HI), 0.5*(DEC_LO+DEC_HI), unit=(u.deg, u.deg))
    t0 = time.time()
    r = v.query_region(pos, radius='1.55 deg', catalog='II/349/ps1')
    print(f'[PS1]   got {len(r[0]):,} rows in {time.time()-t0:.1f}s')
    df = r[0].to_pandas()
    df.rename(columns={'RAJ2000':'ra', 'DEJ2000':'dec'}, inplace=True)
    # box trim
    df = df[(df['ra']>=RA_LO)&(df['ra']<RA_HI)&(df['dec']>=DEC_LO)&(df['dec']<DEC_HI)].reset_index(drop=True)
    print(f'[PS1]   in box: {len(df):,}')
    return df


def pull_sdss(n_tiles=5):
    print(f'[SDSS] tiling box into {n_tiles}×{n_tiles} squares ...')
    from astroquery.sdss import SDSS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    edges_ra  = np.linspace(RA_LO,  RA_HI,  n_tiles+1)
    edges_dec = np.linspace(DEC_LO, DEC_HI, n_tiles+1)
    cell_w = (RA_HI - RA_LO) / n_tiles
    cell_h = (DEC_HI - DEC_LO) / n_tiles
    # radius to cover one tile's diagonal
    cell_r = 0.5 * np.sqrt(cell_w**2 + cell_h**2) * 1.05   # +5% pad
    fields = ['ra','dec','psfMag_u','psfMag_g','psfMag_r','psfMag_i','psfMag_z',
              'psfMagErr_u','psfMagErr_g','psfMagErr_r','psfMagErr_i','psfMagErr_z',
              'type','clean','objid']
    frames = []
    t0 = time.time()
    for i in range(n_tiles):
        for j in range(n_tiles):
            ra0  = 0.5*(edges_ra[i]+edges_ra[i+1])
            dec0 = 0.5*(edges_dec[j]+edges_dec[j+1])
            pos = SkyCoord(ra0, dec0, unit=(u.deg,u.deg))
            try:
                r = SDSS.query_region(pos, radius=f'{cell_r} deg', spectro=False,
                                      photoobj_fields=fields, data_release=16)
                if r is not None and len(r):
                    frames.append(r.to_pandas())
            except Exception as e:
                print(f'[SDSS]   tile ({i},{j}) FAILED: {repr(e)[:60]}')
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f'[SDSS]   {len(df):,} rows (with overlap) in {time.time()-t0:.1f}s')
    if len(df) == 0:
        return df
    df = df.drop_duplicates('objid').reset_index(drop=True)
    df = df[(df['ra']>=RA_LO)&(df['ra']<RA_HI)&(df['dec']>=DEC_LO)&(df['dec']<DEC_HI)].reset_index(drop=True)
    print(f'[SDSS]   {len(df):,} unique in box')
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ps1  = pull_ps1()
    sdss = pull_sdss()
    if len(ps1):
        ps1.to_parquet(OUT/'ps1_cosmos.parquet', index=False)
        ps1.to_csv(OUT/'ps1_cosmos.csv', index=False)
        print(f'[save] ps1_cosmos: {len(ps1):,} × {len(ps1.columns)} → parquet+csv')
    if len(sdss):
        sdss.to_parquet(OUT/'sdss_cosmos.parquet', index=False)
        sdss.to_csv(OUT/'sdss_cosmos.csv', index=False)
        print(f'[save] sdss_cosmos: {len(sdss):,} × {len(sdss.columns)} → parquet+csv')


if __name__ == '__main__':
    main()
