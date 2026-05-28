#!/usr/bin/env python
"""
53_step4_gaia_augment.py — Augment the 4 per-mission Step-3 catalogs with
Gaia DR3 columns.

Strategy (per user, 2026-05-27):
  - Re-query Gaia DR3 over the COSMOS box with the full column set we need.
  - For each mission, propagate every Gaia star to that mission's epoch using
    its OWN proper motion (predicted position = ra_2016 + pmra*Δt / cos δ,
    dec_2016 + pmdec*Δt).
  - Match the propagated Gaia positions against the mission's Step-3 catalog
    within a small ABSOLUTE tolerance.  The tolerance budget covers:
      * Gaia astrometric uncertainty (sub-mas at G<18, growing at faint end)
      * Mission WCS systematic — HST 30-100 mas; JWST / Euclid Gaia-anchored ≈0
      * PM-propagation error: sqrt(σ_pmra² + σ_pmdec²) × Δt
    PM amplitude itself is absorbed by the propagation; the radius does
    NOT have to grow with PM.
  - 4 missions run in parallel (one worker per mission) on this 14-core M4 Pro.

Inputs:
  csvfiles_star/cat_matched_HST.parquet         (208,557 rows)
  csvfiles_star/cat_matched_JWST.parquet        (252,542 rows)
  csvfiles_star/cat_matched_Euclid_VIS.parquet  (721,904 rows)
  csvfiles_star/cat_matched_Euclid_NISP.parquet (667,721 rows)
  csvfiles_star/gaia_dr3_cosmos_v2.parquet      <- re-queried here

Outputs (row counts UNCHANGED from inputs):
  csvfiles_star/cat_matched_HST_with_gaia.parquet
  csvfiles_star/cat_matched_JWST_with_gaia.parquet
  csvfiles_star/cat_matched_Euclid_VIS_with_gaia.parquet
  csvfiles_star/cat_matched_Euclid_NISP_with_gaia.parquet
  + .csv twins
  htmls/star_v01/gaia_match_<mission>.png   (sep vs G, Δmag vs G)
  htmls/star_v01/gaia_augment_summary.txt
"""
from __future__ import annotations
import os
import sys
import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u

warnings.filterwarnings('ignore')

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'star_v01'
HTML.mkdir(parents=True, exist_ok=True)

GAIA_CACHE_V2 = OUT / 'gaia_dr3_cosmos_v2.parquet'

# Mission epochs (from CLAUDE.md and project memory)
MISSION_EPOCH = {
    'HST':         2005.0,   # ACS COSMOS imaging cycle 12-13
    'JWST':        2024.0,   # COSMOS-Web 2023-2024 (use 2024 midpoint)
    'Euclid_VIS':  2024.5,   # Euclid Q1
    'Euclid_NISP': 2024.5,
}

# Matching tolerance per mission (arcsec).  Tightened (2026-05-27 pass 2)
# based on the observed sep distributions from pass 1:
#   HST    : sep<=0.40" captures 99.4% of pass-1 matches
#   JWST   : sep<=0.15" captures 99.3%
#   Euclid_VIS : sep<=0.15" captures 99.1%
#   Euclid_NISP: sep<=0.20" captures 99.1%
# HST is bigger because of the 30-100 mas WCS systematic.  JWST and Euclid
# are Gaia-anchored ⇒ tight.
MATCH_RADIUS_AS = {
    'HST':         0.40,
    'JWST':        0.15,
    'Euclid_VIS':  0.15,
    'Euclid_NISP': 0.20,
}

# Bright-Gaia completeness.  For each mission, every Gaia star with G<18
# whose propagated position falls inside the catalog bbox AND has no
# catalog source within the "confirm" radius is added as a new row
# (source_type='gaia_only_bright', cat_* and DAO columns NaN).
# Confirm radius is ~3-5× the match radius to absorb bright-star
# centroid drift (cores saturate, centroids wander).
GAIA_BRIGHT_THRESHOLD = 18.0
GAIA_BRIGHT_CONFIRM_AS = {
    'HST':         1.5,
    'JWST':        0.6,
    'Euclid_VIS':  0.6,
    'Euclid_NISP': 0.8,
}

GAIA_EPOCH = 2016.0


# ─────────────────────────────────────────────────────────────────────────────
# Gaia query
# ─────────────────────────────────────────────────────────────────────────────

def pull_gaia_v2() -> pd.DataFrame:
    """Re-query Gaia DR3 with the extended column set."""
    if GAIA_CACHE_V2.exists():
        print(f'[gaia] cached → {GAIA_CACHE_V2.name}')
        return pd.read_parquet(GAIA_CACHE_V2)

    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'

    cols = [
        'source_id', 'ra', 'dec', 'ref_epoch',
        # PMs + errors
        'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
        'parallax', 'parallax_error',
        # Photometry + flux_over_error (lets us derive mag errors)
        'phot_g_mean_mag', 'phot_g_mean_flux_over_error',
        'phot_bp_mean_mag', 'phot_bp_mean_flux_over_error',
        'phot_rp_mean_mag', 'phot_rp_mean_flux_over_error',
        # Quality / blending / "ellipticity-like" indicators
        'astrometric_excess_noise', 'astrometric_chi2_al',
        'ruwe',
        'ipd_frac_multi_peak', 'ipd_gof_harmonic_amplitude',
        # Star vs galaxy vs quasar classification
        'classprob_dsc_combmod_star',
        'classprob_dsc_combmod_galaxy',
        'classprob_dsc_combmod_quasar',
        'non_single_star',
    ]
    query = (
        f"SELECT {', '.join(cols)} "
        "FROM gaiadr3.gaia_source "
        "WHERE ra BETWEEN 149.0 AND 151.1 "
        "  AND dec BETWEEN 1.2 AND 3.3 "
        "  AND phot_g_mean_mag < 21.5"
    )
    print('[gaia] submitting extended ADQL query ...')
    t0 = time.time()
    job = Gaia.launch_job_async(query)
    tab = job.get_results()
    df = tab.to_pandas()
    print(f'[gaia]   retrieved {len(df):,} rows in {time.time()-t0:.1f}s')

    # Derive AB-mag-style 1-sigma errors from flux_over_error: σ_mag ≈ 1.0857/SNR
    for band in ('g', 'bp', 'rp'):
        foe = df[f'phot_{band}_mean_flux_over_error'].astype(float).values
        with np.errstate(divide='ignore', invalid='ignore'):
            df[f'phot_{band}_mean_mag_error'] = np.where(
                foe > 0, 1.0857 / foe, np.nan,
            )

    df.to_parquet(GAIA_CACHE_V2, index=False)
    print(f'[gaia]   cached → {GAIA_CACHE_V2.name}')
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Propagation
# ─────────────────────────────────────────────────────────────────────────────

def propagate(gaia: pd.DataFrame, target_epoch: float) -> pd.DataFrame:
    """Propagate Gaia (ra, dec) at ref_epoch (2016.0) to target_epoch using PM.

    pm[ra] is in mas/yr in TRUE-angle units (already times cos(dec)).  So
    Δra (deg) = pmra / cos(dec) × Δt / 3.6e6.
    """
    out = gaia[['source_id']].copy()
    dt = target_epoch - gaia['ref_epoch'].fillna(GAIA_EPOCH).values
    cosd = np.cos(np.deg2rad(gaia['dec'].values))
    pmra  = gaia['pmra'].fillna(0.0).values
    pmdec = gaia['pmdec'].fillna(0.0).values
    out['ra_prop']  = gaia['ra'].values  + (pmra / cosd) * dt / 3.6e6
    out['dec_prop'] = gaia['dec'].values + pmdec * dt / 3.6e6
    # PM-propagation 1-sigma uncertainty along each axis (arcsec)
    pmra_e  = gaia['pmra_error'].fillna(0.0).values
    pmdec_e = gaia['pmdec_error'].fillna(0.0).values
    out['prop_err_ra_as']  = np.abs(pmra_e  * dt) / 1000.0
    out['prop_err_dec_as'] = np.abs(pmdec_e * dt) / 1000.0
    out['dt_yr'] = dt
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-mission worker
# ─────────────────────────────────────────────────────────────────────────────

GAIA_COLS_TO_CARRY = [
    'source_id', 'ra', 'dec', 'ref_epoch',
    'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
    'parallax', 'parallax_error',
    'phot_g_mean_mag',  'phot_g_mean_mag_error',
    'phot_bp_mean_mag', 'phot_bp_mean_mag_error',
    'phot_rp_mean_mag', 'phot_rp_mean_mag_error',
    'astrometric_excess_noise', 'ruwe',
    'ipd_frac_multi_peak', 'ipd_gof_harmonic_amplitude',
    'classprob_dsc_combmod_star',
    'classprob_dsc_combmod_galaxy',
    'classprob_dsc_combmod_quasar',
    'non_single_star',
]


def match_mission(mission: str) -> dict:
    """Worker — load Step-3 catalog, propagate Gaia, match, write outputs."""
    t0 = time.time()
    print(f'[{mission}] starting ...')

    # Inputs
    cat = pd.read_parquet(OUT / f'cat_matched_{mission}.parquet')
    gaia_full = pd.read_parquet(GAIA_CACHE_V2)
    epoch = MISSION_EPOCH[mission]
    radius_as = MATCH_RADIUS_AS[mission]

    # Propagate Gaia to this mission's epoch
    prop = propagate(gaia_full, epoch)
    print(f'[{mission}]   cat={len(cat):,}, gaia={len(gaia_full):,}, '
          f'epoch={epoch}, radius={radius_as}"')

    # Cross-match
    cg = SkyCoord(prop['ra_prop'].values * u.deg,
                  prop['dec_prop'].values * u.deg)
    cc = SkyCoord(cat['cat_ra'].values * u.deg,
                  cat['cat_dec'].values * u.deg)
    idx_g, idx_c, sep, _ = search_around_sky(cg, cc, radius_as * u.arcsec)
    print(f'[{mission}]   pairs within {radius_as}": {len(idx_g):,}')

    # For each catalog row, keep the closest Gaia within radius
    sep_as = sep.arcsec
    # Sort pairs by sep, then take first hit per catalog idx (= closest)
    order = np.argsort(sep_as)
    idx_g = idx_g[order]
    idx_c = idx_c[order]
    sep_as = sep_as[order]

    seen = np.zeros(len(cat), dtype=bool)
    best_g = np.full(len(cat), -1, dtype=np.int64)
    best_sep = np.full(len(cat), np.nan)
    for ig, ic, s in zip(idx_g, idx_c, sep_as):
        if not seen[ic]:
            seen[ic] = True
            best_g[ic] = ig
            best_sep[ic] = s

    n_matched = int(seen.sum())
    print(f'[{mission}]   unique catalog rows matched: {n_matched:,} '
          f'({100*n_matched/len(cat):.2f}% of catalog)')

    # Build the Gaia-augmented columns aligned to cat row order
    out = cat.copy()
    out['gaia_sep_as']    = best_sep
    out['gaia_dt_yr']     = np.where(best_g >= 0, prop['dt_yr'].values[best_g], np.nan)
    out['gaia_ra_prop']   = np.where(best_g >= 0, prop['ra_prop'].values[best_g], np.nan)
    out['gaia_dec_prop']  = np.where(best_g >= 0, prop['dec_prop'].values[best_g], np.nan)
    out['gaia_prop_err_ra_as']  = np.where(best_g >= 0, prop['prop_err_ra_as'].values[best_g],  np.nan)
    out['gaia_prop_err_dec_as'] = np.where(best_g >= 0, prop['prop_err_dec_as'].values[best_g], np.nan)
    for col in GAIA_COLS_TO_CARRY:
        vals = gaia_full[col].values
        if vals.dtype.kind in ('U', 'O'):
            # object / string column — must handle NaN explicitly
            out[f'gaia_{col}'] = np.where(best_g >= 0,
                                          vals[np.where(best_g >= 0, best_g, 0)],
                                          None)
        else:
            out[f'gaia_{col}'] = np.where(best_g >= 0,
                                          vals[np.where(best_g >= 0, best_g, 0)],
                                          np.nan)

    # source_type tag for existing rows
    out['source_type'] = np.where(seen, 'catalog+gaia', 'catalog')

    # ── Bright-Gaia completeness pass ──────────────────────────────────────
    # Find Gaia G<18 stars in the mission's catalog bbox that the catalog
    # missed entirely (no catalog source within the confirm radius).
    confirm_as = GAIA_BRIGHT_CONFIRM_AS[mission]
    ra_min, ra_max = float(cat['cat_ra'].min()), float(cat['cat_ra'].max())
    dec_min, dec_max = float(cat['cat_dec'].min()), float(cat['cat_dec'].max())

    gmag = gaia_full['phot_g_mean_mag'].values
    bright_mask = (
        (gmag < GAIA_BRIGHT_THRESHOLD)
        & (prop['ra_prop'].values  >= ra_min) & (prop['ra_prop'].values  <= ra_max)
        & (prop['dec_prop'].values >= dec_min) & (prop['dec_prop'].values <= dec_max)
    )
    n_bright_in_bbox = int(bright_mask.sum())
    print(f'[{mission}]   Gaia G<{GAIA_BRIGHT_THRESHOLD} in bbox: {n_bright_in_bbox:,}')

    # For each bright Gaia in bbox, is there ANY cat source within confirm radius?
    bright_idx = np.where(bright_mask)[0]
    if len(bright_idx):
        cg_bright = SkyCoord(prop['ra_prop'].values[bright_idx]  * u.deg,
                             prop['dec_prop'].values[bright_idx] * u.deg)
        ib_g, _, _, _ = search_around_sky(cg_bright, cc, confirm_as * u.arcsec)
        has_neighbour = np.zeros(len(bright_idx), dtype=bool)
        has_neighbour[np.unique(ib_g)] = True
        gaia_only_idx = bright_idx[~has_neighbour]
    else:
        gaia_only_idx = np.array([], dtype=np.int64)

    print(f'[{mission}]   Gaia G<{GAIA_BRIGHT_THRESHOLD} missing from catalog '
          f'(no source within {confirm_as}"): {len(gaia_only_idx):,}')

    # Build the gaia_only_bright rows.  All cat_*/DAO columns are NaN.
    if len(gaia_only_idx) > 0:
        extra = pd.DataFrame(index=range(len(gaia_only_idx)),
                             columns=out.columns, dtype=object)
        # Numeric columns → NaN; object columns → None
        for col in out.columns:
            if out[col].dtype.kind in ('f', 'i', 'u', 'b'):
                extra[col] = np.nan
            else:
                extra[col] = None
        # Populate gaia_* columns from the unmatched bright Gaia
        extra['gaia_sep_as']    = np.nan
        extra['gaia_dt_yr']     = prop['dt_yr'].values[gaia_only_idx]
        extra['gaia_ra_prop']   = prop['ra_prop'].values[gaia_only_idx]
        extra['gaia_dec_prop']  = prop['dec_prop'].values[gaia_only_idx]
        extra['gaia_prop_err_ra_as']  = prop['prop_err_ra_as'].values[gaia_only_idx]
        extra['gaia_prop_err_dec_as'] = prop['prop_err_dec_as'].values[gaia_only_idx]
        for col in GAIA_COLS_TO_CARRY:
            extra[f'gaia_{col}'] = gaia_full[col].values[gaia_only_idx]
        extra['source_type'] = 'gaia_only_bright'

        # Coerce dtypes back to match `out` (the index-and-object construction
        # leaves everything as object, which would break parquet writes)
        extra = extra.astype({c: out[c].dtype for c in out.columns
                              if out[c].dtype.kind in ('f', 'i', 'u')},
                             errors='ignore')

        out = pd.concat([out, extra], ignore_index=True)

    # Write
    out_parq = OUT / f'cat_matched_{mission}_with_gaia.parquet'
    out_csv  = OUT / f'cat_matched_{mission}_with_gaia.csv'
    out.to_parquet(out_parq, index=False)
    out.to_csv(out_csv, index=False)
    n_cat_gaia = int((out['source_type'] == 'catalog+gaia').sum())
    n_cat_only = int((out['source_type'] == 'catalog').sum())
    n_gaia_only = int((out['source_type'] == 'gaia_only_bright').sum())
    print(f'[{mission}]   wrote {out_parq.name} '
          f'({len(out):,} rows × {len(out.columns)} cols) — '
          f'catalog={n_cat_only:,}, catalog+gaia={n_cat_gaia:,}, '
          f'gaia_only_bright={n_gaia_only:,}')

    # Diagnostic plot — sep vs G mag and Δmag vs G mag
    diag_path = HTML / f'gaia_match_{mission}.png'
    _make_diag_plot(out, mission, diag_path)

    # Summary stats
    elapsed = time.time() - t0
    return {
        'mission': mission,
        'n_cat': len(cat),
        'n_matched': n_matched,
        'frac_matched': n_matched / len(cat),
        'radius_as': radius_as,
        'confirm_as': confirm_as,
        'n_bright_in_bbox': n_bright_in_bbox,
        'n_gaia_only_bright': len(gaia_only_idx),
        'n_total_out': len(out),
        'epoch': epoch,
        'elapsed_s': elapsed,
        'output_parquet': str(out_parq),
        'diag_png': str(diag_path),
    }


def _make_diag_plot(df: pd.DataFrame, mission: str, path: Path) -> None:
    g = df.dropna(subset=['gaia_sep_as', 'gaia_phot_g_mean_mag']).copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: sep vs G
    ax = axes[0]
    ax.scatter(g['gaia_phot_g_mean_mag'], g['gaia_sep_as']*1000.0,
               s=4, alpha=0.3, color='C0', rasterized=True)
    ax.set_xlabel('Gaia G mag')
    ax.set_ylabel('match separation (mas)')
    ax.set_title(f'{mission}: separation vs G  (N={len(g):,})')
    ax.grid(alpha=0.3)

    # Right: Δmag vs G (mission-band − G).  Pick a sensible band per mission.
    band = {
        'HST': 'F814W',
        'JWST': 'F115W',
        'Euclid_VIS': 'VIS',
        'Euclid_NISP': 'NIR_H',
    }[mission]
    magcol = f'cat_mag_{band}'
    if magcol in g.columns:
        dm = g[magcol] - g['gaia_phot_g_mean_mag']
        ax = axes[1]
        ax.scatter(g['gaia_phot_g_mean_mag'], dm,
                   s=4, alpha=0.3, color='C1', rasterized=True)
        ax.set_xlabel('Gaia G mag')
        ax.set_ylabel(f'cat_mag_{band} − G')
        ax.set_title(f'{mission}: colour vs G  (band={band})')
        ax.grid(alpha=0.3)

    fig.suptitle(f'{mission} — Gaia DR3 match diagnostics', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main — parallel dispatch
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print('=' * 70)
    print('Step 4 — Gaia DR3 augmentation of 4 per-mission catalogs')
    print('=' * 70)

    # Single Gaia query (cached) before fanning out
    gaia = pull_gaia_v2()
    print(f'[gaia] {len(gaia):,} stars, '
          f'G={gaia["phot_g_mean_mag"].min():.2f}–'
          f'{gaia["phot_g_mean_mag"].max():.2f}')

    missions = ['HST', 'JWST', 'Euclid_VIS', 'Euclid_NISP']

    # Spawn 4 worker processes (one per mission)
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(match_mission, m): m for m in missions}
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f'[{m}] FAILED: {e!r}')
                import traceback
                traceback.print_exc()
                raise

    # Summary
    print()
    print('=' * 70)
    print('Summary')
    print('=' * 70)
    summary_lines = []
    summary_lines.append(
        f'{"mission":<13} {"r_match":>7} {"r_conf":>7} {"epoch":>7} '
        f'{"n_cat":>9} {"matched":>8} {"G<18bbox":>9} {"gaia_only":>10} '
        f'{"n_total":>9} {"time":>6}'
    )
    summary_lines.append('-' * 100)
    for r in sorted(results, key=lambda x: x['mission']):
        summary_lines.append(
            f'{r["mission"]:<13} {r["radius_as"]:>6.2f}" '
            f'{r["confirm_as"]:>6.2f}" {r["epoch"]:>7.1f} '
            f'{r["n_cat"]:>9,} {r["n_matched"]:>8,} '
            f'{r["n_bright_in_bbox"]:>9,} {r["n_gaia_only_bright"]:>10,} '
            f'{r["n_total_out"]:>9,} {r["elapsed_s"]:>5.1f}s'
        )
    text = '\n'.join(summary_lines)
    print(text)
    summary_path = HTML / 'gaia_augment_summary.txt'
    summary_path.write_text(text + '\n')
    print(f'\nWall time: {time.time() - t_total:.1f}s')
    print(f'Summary → {summary_path}')


if __name__ == '__main__':
    main()
