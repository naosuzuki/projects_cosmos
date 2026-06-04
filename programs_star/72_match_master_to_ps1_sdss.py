#!/usr/bin/env python
"""
72_match_master_to_ps1_sdss.py — find PS1 + SDSS counterparts for each
master 4-way star, using Gaia DR3 PM to propagate positions to each
survey's mean epoch.

Master input:  csvfiles_star/master_stars_4way.parquet  (9,699 sources)
PS1 input:     csvfiles_star/ps1_cosmos.parquet
SDSS input:    csvfiles_star/sdss_cosmos.parquet

Epoch handling:
  Gaia reference epoch         : 2016.0
  PS1 3π mean epoch in COSMOS  : 2012.0
  SDSS imaging mean in COSMOS  : 2005.5
For each master star with Gaia DR3 PM, the position is propagated from
its Gaia 2016 RA/Dec by pmra/pmdec to the target survey epoch (this is
the same code used in step 4 of the master pipeline).  Stars without
Gaia PM use the JWST cat_ra/cat_dec (epoch ~2024) as the search anchor,
with no PM correction.

Match radius:
  Gaia-anchored stars      : 0.5″
  Non-Gaia stars (no PM)   : 2.0″
Tie-break (multiple candidates within the cone): pick the one whose
PS1 i / SDSS i magnitude is closest to our HST F814W (within 1.0 mag);
fall back to the closest in separation if no candidate passes the mag
sanity.  AGN/QSO get a looser mag tolerance (2.0 mag) because of
variability.

Output:
  csvfiles_star/master_stars_4way_with_ps1_sdss.parquet  (+ .csv)
Columns appended:
  ps1_objID, ps1_{g,r,i,z,y}mag + e_*, ps1_sep_arcsec, ps1_propagated
  sdss_objid, sdss_psfMag_{u,g,r,i,z} + Err, sdss_type, sdss_sep_arcsec, sdss_propagated
"""
from __future__ import annotations
import time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u

warnings.filterwarnings('ignore')
ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'

GAIA_EPOCH = 2016.0
PS1_EPOCH  = 2012.0
SDSS_EPOCH = 2005.5
RAD_GAIA   = 0.5     # arcsec
RAD_NOGAIA = 2.0
MAG_TOL_STAR = 1.0
MAG_TOL_AGN  = 2.0


def propagate(ra, dec, pmra, pmdec, dt_yr):
    """Propagate Gaia (RA, Dec) deg by (pmra, pmdec) mas/yr over dt_yr."""
    cosd = np.cos(np.deg2rad(dec))
    ra_new  = ra  + (pmra  / cosd) * dt_yr / 3.6e6
    dec_new = dec + pmdec           * dt_yr / 3.6e6
    return ra_new, dec_new


def make_anchor(master, target_epoch):
    """Best position for each master star at the target epoch.
    Uses Gaia propagation where pmra/pmdec are finite; otherwise the
    JWST catalogue position (epoch ~2024, no PM correction).
    Returns: ra, dec (arrays, deg), used_gaia (bool array)."""
    gpmra  = master['gaia_pmra'].values
    gpmdec = master['gaia_pmdec'].values
    gra    = master['gaia_ra'].values
    gdec   = master['gaia_dec'].values
    use_gaia = np.isfinite(gpmra) & np.isfinite(gpmdec) & np.isfinite(gra) & np.isfinite(gdec)
    dt = target_epoch - GAIA_EPOCH
    ra_p, dec_p = propagate(gra, gdec, np.where(use_gaia, gpmra, 0.0),
                             np.where(use_gaia, gpmdec, 0.0), dt)
    # fallback: JWST cat position (2024)
    ra_fb  = master['cat_ra_jwst'].values
    dec_fb = master['cat_dec_jwst'].values
    ra  = np.where(use_gaia, ra_p,  ra_fb)
    dec = np.where(use_gaia, dec_p, dec_fb)
    return ra, dec, use_gaia


def find_match(master, surv, ra_m, dec_m, used_gaia, surv_mag_col, our_mag, is_agn):
    """For each master row, pick best surv counterpart.
    Returns row indices into surv (or -1) and separation in arcsec.

    Logic:
      1. For each master row, run search_around_sky with the per-row
         radius (RAD_GAIA if Gaia-anchored, else RAD_NOGAIA).
      2. For multiple candidates: prefer mag-consistent
         (|surv_mag - our_mag| < MAG_TOL_STAR / MAG_TOL_AGN).
      3. Else pick closest by sep.
    Implementation: one pass at the LOOSER radius, then per-row filter.
    """
    cm = SkyCoord(ra_m * u.deg, dec_m * u.deg)
    cs = SkyCoord(surv['ra'].values * u.deg, surv['dec'].values * u.deg)
    idx_m, idx_s, sep, _ = search_around_sky(cm, cs, RAD_NOGAIA * u.arcsec)
    sep_as = sep.arcsec
    surv_mags = surv[surv_mag_col].values

    best = np.full(len(master), -1, dtype=np.int64)
    best_sep = np.full(len(master), np.nan)

    # group candidates by master row
    order = np.argsort(idx_m, kind='stable')
    idx_m = idx_m[order]; idx_s = idx_s[order]; sep_as = sep_as[order]
    # find run starts for each master row index
    if len(idx_m) == 0:
        return best, best_sep
    boundaries = np.concatenate(([0], np.where(np.diff(idx_m) != 0)[0] + 1, [len(idx_m)]))
    for k in range(len(boundaries) - 1):
        s, e = boundaries[k], boundaries[k+1]
        mrow = int(idx_m[s])
        # enforce per-row tight radius if Gaia-anchored
        max_r = RAD_GAIA if used_gaia[mrow] else RAD_NOGAIA
        keep = sep_as[s:e] <= max_r
        if not keep.any():
            continue
        cand_idx_s = idx_s[s:e][keep]
        cand_sep   = sep_as[s:e][keep]
        # mag check
        our_m = our_mag[mrow]
        mag_tol = MAG_TOL_AGN if is_agn[mrow] else MAG_TOL_STAR
        if np.isfinite(our_m):
            cand_mag = surv_mags[cand_idx_s]
            ok = np.isfinite(cand_mag) & (np.abs(cand_mag - our_m) < mag_tol)
            if ok.any():
                # among mag-consistent: pick closest
                j = np.argmin(np.where(ok, cand_sep, np.inf))
                best[mrow] = int(cand_idx_s[j]); best_sep[mrow] = float(cand_sep[j])
                continue
        # fallback: closest in separation regardless of mag
        j = np.argmin(cand_sep)
        best[mrow] = int(cand_idx_s[j]); best_sep[mrow] = float(cand_sep[j])
    return best, best_sep


def main():
    t0 = time.time()
    print('Loading inputs ...')
    master = pd.read_parquet(OUT / 'master_stars_4way.parquet')
    ps1    = pd.read_parquet(OUT / 'ps1_cosmos.parquet')
    sdss   = pd.read_parquet(OUT / 'sdss_cosmos.parquet')
    print(f'  master: {len(master):,}   PS1: {len(ps1):,}   SDSS: {len(sdss):,}')

    # our reference HST F814W mag for sanity
    our_F814 = master['cat_mag_F814W_hst'].values
    is_agn = master['is_agn_qso'].astype(bool).values

    # ── PS1 match ──────────────────────────────────────────────────────────
    print('Propagating Gaia → PS1 epoch ...')
    ra_ps1, dec_ps1, used_gaia_ps1 = make_anchor(master, PS1_EPOCH)
    n_gaia = int(used_gaia_ps1.sum())
    print(f'  Gaia-propagated: {n_gaia:,}   non-Gaia (JWST 2024): {len(master)-n_gaia:,}')
    print('Matching to PS1 ...')
    idx_ps, sep_ps = find_match(master, ps1, ra_ps1, dec_ps1, used_gaia_ps1,
                                 'imag', our_F814, is_agn)
    nm_ps = int((idx_ps >= 0).sum())
    print(f'  PS1 matches: {nm_ps:,} / {len(master):,} ({100*nm_ps/len(master):.1f}%)')

    # ── SDSS match ─────────────────────────────────────────────────────────
    print('Propagating Gaia → SDSS epoch ...')
    ra_sd, dec_sd, used_gaia_sd = make_anchor(master, SDSS_EPOCH)
    print('Matching to SDSS ...')
    idx_sd, sep_sd = find_match(master, sdss, ra_sd, dec_sd, used_gaia_sd,
                                 'psfMag_i', our_F814, is_agn)
    nm_sd = int((idx_sd >= 0).sum())
    print(f'  SDSS matches: {nm_sd:,} / {len(master):,} ({100*nm_sd/len(master):.1f}%)')

    # ── append columns to master ───────────────────────────────────────────
    out = master.copy()
    # PS1
    ps1_cols = {'objID':'ps1_objID',
                'gmag':'ps1_gmag','e_gmag':'ps1_e_gmag',
                'rmag':'ps1_rmag','e_rmag':'ps1_e_rmag',
                'imag':'ps1_imag','e_imag':'ps1_e_imag',
                'zmag':'ps1_zmag','e_zmag':'ps1_e_zmag',
                'ymag':'ps1_ymag','e_ymag':'ps1_e_ymag'}
    for src, dst in ps1_cols.items():
        arr = np.full(len(out), np.nan)
        mask = idx_ps >= 0
        vals = ps1[src].values[idx_ps[mask].astype(int)]
        if vals.dtype.kind in ('f','i','u'):
            arr[mask] = vals
        else:
            arr = np.full(len(out), None, dtype=object); arr[mask] = vals
        out[dst] = arr
    out['ps1_sep_arcsec'] = sep_ps
    out['ps1_propagated_with_gaia'] = used_gaia_ps1
    # SDSS
    sdss_cols = {'objid':'sdss_objid','type':'sdss_type',
                 'psfMag_u':'sdss_psfMag_u','psfMag_g':'sdss_psfMag_g',
                 'psfMag_r':'sdss_psfMag_r','psfMag_i':'sdss_psfMag_i',
                 'psfMag_z':'sdss_psfMag_z',
                 'psfMagErr_u':'sdss_psfMagErr_u','psfMagErr_g':'sdss_psfMagErr_g',
                 'psfMagErr_r':'sdss_psfMagErr_r','psfMagErr_i':'sdss_psfMagErr_i',
                 'psfMagErr_z':'sdss_psfMagErr_z'}
    for src, dst in sdss_cols.items():
        arr = np.full(len(out), np.nan)
        mask = idx_sd >= 0
        vals = sdss[src].values[idx_sd[mask].astype(int)]
        if vals.dtype.kind in ('f','i','u'):
            arr[mask] = vals
        else:
            arr = np.full(len(out), None, dtype=object); arr[mask] = vals
        out[dst] = arr
    out['sdss_sep_arcsec'] = sep_sd
    out['sdss_propagated_with_gaia'] = used_gaia_sd

    out_pq = OUT / 'master_stars_4way_with_ps1_sdss.parquet'
    out_cs = OUT / 'master_stars_4way_with_ps1_sdss.csv'
    out.to_parquet(out_pq, index=False)
    out.to_csv(out_cs, index=False)
    print(f'\nWrote {out_pq.name}: {len(out):,} rows × {len(out.columns)} cols')

    # ── breakdown by source class & by Gaia anchoring ──────────────────────
    print('\nMatch-rate breakdown:')
    print(f'{"sample":<24} {"PS1":>10} {"SDSS":>10}')
    print('-' * 50)
    for label, mask in [
        ('All',                    np.ones(len(out), dtype=bool)),
        ('stars',                  ~is_agn),
        ('AGN/QSO',                is_agn),
        ('Gaia-anchored (PS1)',    used_gaia_ps1),
        ('non-Gaia (PS1)',         ~used_gaia_ps1),
    ]:
        pn = int((idx_ps[mask] >= 0).sum())
        sn = int((idx_sd[mask] >= 0).sum())
        n  = int(mask.sum())
        if n == 0: continue
        print(f'{label:<24} {pn:>5,}/{n:<5,}  {sn:>5,}/{n:<5,}')

    print(f'\nWall time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
