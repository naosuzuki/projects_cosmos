"""
v03 regression tests — one short check per documented v01 bug.

Run with:
    pytest tests/test_v03_regression.py -v

These tests target the four known v01/v02 issues recorded in
`programs_star/CLAUDE.md` §"Known bugs":
    B1 — primary_id not globally unique across overlapping HST tiles
    B3 — saturated bright stars dropped from HST catalog (no PM)
    B5 — pmtot understates motion when raw cat-ra delta is large
    B6 — PM joined by hst_id alone (per-tile) → values copy across rows

Each test:
  * SKIPS cleanly if v03 master is not built yet.
  * FAILS on the documented bug pattern (so the fix can be verified).
  * Comes with a docstring that says what's being checked and why.

The companion v01-baseline tests (test_*_baseline_v01) demonstrate the
SAME bug pattern in the existing v01 master, so the user can see by how
much v03 needs to improve.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


# ============================================================
# B1 — primary_id globally unique
# ============================================================

def test_b1_primary_id_globally_unique(v03_master):
    """B1: every row of the master should have a unique primary_id.

    v01 violated this because primary_id = "HSTID_<hst_id>" and hst_id is
    per-tile (1..N per HST tile), so the SAME hst_id maps to different
    physical sources across overlapping tiles.  In v01, HSTID_192
    appeared 121 times.
    """
    n_total   = len(v03_master)
    n_unique  = v03_master['primary_id'].nunique()
    dup_ratio = n_total / n_unique if n_unique else float('inf')

    msg = (f'{n_total - n_unique} duplicate primary_ids '
           f'(dup ratio {dup_ratio:.2f}× — should be 1.00)')
    assert v03_master['primary_id'].is_unique, msg


def test_b1_baseline_v01(v01_master):
    """Document the v01 magnitude of B1 — should fail by design."""
    n_total  = len(v01_master)
    n_unique = v01_master['primary_id'].nunique()
    dup_ratio = n_total / n_unique
    # In v01 the dup ratio is ~29× for HST-prefix rows.  Whole master ~1.04.
    if v01_master['primary_id'].is_unique:
        pytest.skip('v01 master shows no B1 collisions — unexpected, please inspect')
    pytest.xfail(f'v01 baseline: {n_total - n_unique} dup primary_ids '
                 f'(ratio {dup_ratio:.2f}×) — documents the B1 bug magnitude')


# ============================================================
# B3 — saturated Gaia bright stars must be present in master
# ============================================================

def test_b3_bright_gaia_stars_present(v03_master):
    """B3: every Gaia G < 18 star in the COSMOS box should have either an
    HST detection (cat_ra_hst is not NaN) OR an is_saturated_hst=True
    flag.  v01 silently dropped saturated stars: no row, no flag, no PM.
    """
    if 'gaia_g_mag' not in v03_master.columns:
        pytest.skip('v03 schema missing gaia_g_mag; cannot test B3')

    bright = v03_master[v03_master['gaia_g_mag'] < 18]
    has_hst = bright['cat_ra_hst'].notna()
    sat_flag_col = 'is_saturated_hst' if 'is_saturated_hst' in v03_master.columns else None

    if sat_flag_col is None:
        # If v03 doesn't carry an is_saturated_hst column, every bright
        # Gaia star MUST have an HST detection.
        n_missing = (~has_hst).sum()
        assert n_missing == 0, (
            f'{n_missing} Gaia G<18 stars have no HST detection and no '
            f'is_saturated_hst flag — B3 fix incomplete')
    else:
        sat = bright[sat_flag_col].fillna(False)
        n_unaccounted = ((~has_hst) & (~sat)).sum()
        assert n_unaccounted == 0, (
            f'{n_unaccounted} Gaia G<18 stars are missing both HST '
            f'detection and is_saturated_hst flag')


def test_b3_baseline_v01(v01_master):
    """Document the v01 magnitude of B3."""
    if 'gaia_g_mag' not in v01_master.columns:
        pytest.skip('v01 schema missing gaia_g_mag; cannot show B3 magnitude')
    bright = v01_master[v01_master['gaia_g_mag'] < 18]
    n_no_hst = bright['cat_ra_hst'].isna().sum()
    pytest.xfail(f'v01 baseline: {n_no_hst} of {len(bright)} Gaia G<18 stars '
                 f'have no HST detection (and no saturation flag in v01)')


# ============================================================
# B5 — pmtot consistent with raw cat-ra/dec delta
# ============================================================

def _raw_pm_mas_per_yr(df, ra1, dec1, ra2, dec2, baseline_yr):
    """Compute raw on-sky PM from raw catalog (RA, Dec) at two epochs."""
    cosd = np.cos(np.deg2rad((df[dec1] + df[dec2]).values / 2))
    dra  = (df[ra2].values  - df[ra1].values)  * cosd * 3.6e6   # mas
    ddec = (df[dec2].values - df[dec1].values)            * 3.6e6   # mas
    return np.hypot(dra, ddec) / baseline_yr


def test_b5_pmtot_agrees_with_raw_delta_hst_jwst(v03_master):
    """B5: For sources detected in both HST and JWST, the PM stored in
    pmtot_HST_JWST should agree with the raw (cat_ra/cat_dec) delta to
    within 3σ, for ≥99 % of sources.  v01 showed cases where stored PM
    was 5–10× smaller than raw delta implied (centroid mismatch bug).
    """
    m = v03_master
    have_both = m[m['cat_ra_hst'].notna() & m['cat_ra_jwst'].notna()
                  & m['pmtot_HST_JWST'].notna()
                  & m['pmtot_err_HST_JWST'].notna()]
    if len(have_both) < 100:
        pytest.skip(f'only {len(have_both)} 2-epoch pairs available — too few')

    raw = _raw_pm_mas_per_yr(have_both, 'cat_ra_hst', 'cat_dec_hst',
                                       'cat_ra_jwst', 'cat_dec_jwst',
                                       baseline_yr=20.0)
    diff = np.abs(have_both['pmtot_HST_JWST'].values - raw)
    err  = have_both['pmtot_err_HST_JWST'].values
    within_3sigma = diff <= 3 * err

    frac_ok = within_3sigma.mean()
    assert frac_ok >= 0.99, (
        f'only {frac_ok*100:.1f}% of HST↔JWST PMs agree with raw delta '
        f'within 3σ — B5 fix incomplete')


def test_b5_baseline_v01(v01_master):
    """Document the v01 magnitude of B5."""
    m = v01_master
    have_both = m[m['cat_ra_hst'].notna() & m['cat_ra_jwst'].notna()
                  & m['pmtot_HST_JWST'].notna()
                  & m['pmtot_err_HST_JWST'].notna()]
    if len(have_both) < 100:
        pytest.skip(f'only {len(have_both)} pairs in v01 — too few to characterise')
    raw = _raw_pm_mas_per_yr(have_both, 'cat_ra_hst', 'cat_dec_hst',
                                       'cat_ra_jwst', 'cat_dec_jwst',
                                       baseline_yr=20.0)
    diff = np.abs(have_both['pmtot_HST_JWST'].values - raw)
    err  = have_both['pmtot_err_HST_JWST'].values
    frac_ok = (diff <= 3 * err).mean()
    pytest.xfail(f'v01 baseline: only {frac_ok*100:.1f}% of HST↔JWST PMs '
                 f'agree with raw delta within 3σ (B5)')


# ============================================================
# B6 — PM join uses (tile, hst_id) tuple, not hst_id alone
# ============================================================

def test_b6_pm_join_uses_tile_tuple(v03_master):
    """B6: the master must carry an hst_tile column so PM tables can join
    on (hst_tile, hst_id) instead of hst_id alone.  In v01 the master
    has no hst_tile, so the same PM value gets copied to every row that
    shares an hst_id across overlapping tiles.
    """
    schema_ok = 'hst_tile' in v03_master.columns
    assert schema_ok, (
        'master is missing hst_tile column — PM join cannot disambiguate '
        'sources with the same hst_id across overlapping HST tiles')

    # And the (hst_tile, hst_id) tuple must be unique among rows that have HST
    have_hst = v03_master.dropna(subset=['hst_id', 'hst_tile'])
    if len(have_hst) == 0:
        pytest.skip('no HST-detected rows yet')
    n_dup = have_hst.duplicated(['hst_tile', 'hst_id'], keep=False).sum()
    assert n_dup == 0, f'{n_dup} rows share (hst_tile, hst_id) — should be 0'


def test_b6_baseline_v01(v01_master):
    """v01 has no hst_tile column at all — that IS the B6 bug."""
    if 'hst_tile' in v01_master.columns:
        pytest.skip('v01 unexpectedly has hst_tile')
    # Quantify: how many distinct hst_ids appear multiple times in master?
    have_hst = v01_master.dropna(subset=['hst_id'])
    counts = have_hst['hst_id'].value_counts()
    n_colliding = (counts > 1).sum()
    n_rows_in_collisions = counts[counts > 1].sum()
    pytest.xfail(f'v01 baseline: no hst_tile column; {n_colliding} hst_ids '
                 f'have collisions affecting {n_rows_in_collisions} rows (B6)')
