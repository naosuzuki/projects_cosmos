#!/usr/bin/env python
"""
55_step6_refine_pointsource.py — Step 6: refine each cross-matched catalog
with a transparent per-magnitude-bin point-source flag.

Training set ("true point sources"):
  - Gaia DR3 stars: classprob_dsc_combmod_star > 0.9 AND ruwe < 1.4
  - COSMOS-Web AGN/QSO: LePhare type==2 OR flag_chandra==1 OR HSC flag_star_hsc
  - COSMOS-Web stars:   LePhare type==1 OR flag_star
("True extended sources" / negatives):
  - COSMOS-Web LePhare type==0 (galaxy) AND NOT star/AGN flag

Per-mission morphology features used (whatever the catalog provides):
  HST   : cat_fwhm_image_as, cat_a_image_as, cat_b_image_as, ellip,
          cat_class_star, cat_mu_class, DAO F814W sharp/round
  JWST  : cat_fwhm_as, cat_sersic_n, cat_axratio, cat_radius_sersic,
          cat_flag_star, cat_flag_blend, DAO 4-band sharp/round
  VIS   : cat_fwhm_pix, cat_ellipticity, cat_phz_classification,
          cat_point_like_prob, DAO sharp/round
  NISP  : same as VIS

For each mission and each 1-mag bin (in that mission's primary band):
  - Threshold = 99th percentile of feature value among training positives.
  - A source passes mission M's point-source test if all relevant
    morphology features are below their per-mag-bin thresholds (handled
    softly — a source can have ≥N of M criteria passing).

Per-row flags written to each of the 9 cross-matched catalogs:
  is_point_source_<m>             bool per mission (True if mission says PS)
  n_criteria_<m>                  int  number of mission-m criteria passed
  is_point_source                 bool master flag: any mission says PS
                                       OR star is gaia_anchored star
                                       OR cluster is gaia_only_bright
  point_source_evidence           comma-separated list of evidence strings
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
from astropy.io import fits

warnings.filterwarnings('ignore')

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'star_v01'
HTML.mkdir(parents=True, exist_ok=True)

CW_PATH = '/Volumes/exdisk1/data/catalog/JWST/COSMOS/COSMOSWeb_mastercatalog_v1.1.fits'

TAG = {
    'HST':         'hst',
    'JWST':        'jwst',
    'Euclid_VIS':  'vis',
    'Euclid_NISP': 'nisp',
}

# Magnitude column to use for binning per mission (the "primary" band)
MAG_COL = {
    'HST':         'cat_mag_F814W',
    'JWST':        'cat_mag_F115W',
    'Euclid_VIS':  'cat_mag_VIS',
    'Euclid_NISP': 'cat_mag_NIR_H',
}

# Per-mission discriminating morphology feature.  Each is the ONE feature
# we trust to separate point sources from galaxies at non-saturated mags.
# For HST we compute ellip = 1 - b/a from (a_image, b_image).  For JWST
# we use 1 - cat_axratio.  For Euclid VIS/NISP we use cat_ellipticity.
MORPH_ELLIP_FEATURE = {
    'HST':         'ellip',             # derived from a/b image
    'JWST':        'ellip',             # derived from cat_axratio
    'Euclid_VIS':  'cat_ellipticity',   # native
    'Euclid_NISP': 'cat_ellipticity',   # native
}

# Per-mission saturation cutoff: BRIGHTER than this mag, morphology is
# unreliable (saturation cores, diffraction spikes), and we fall back to
# catalog/Gaia flags alone.
SATURATION_MAG = {
    'HST':         20.0,    # F814W
    'JWST':        20.0,    # F115W (COSMOS-Web depth)
    'Euclid_VIS':  20.0,
    'Euclid_NISP': 18.0,
}

# Per-mission "catalog says it's a point source" flag — STRONG evidence
PRIMARY_PS_FLAG = {
    'HST': lambda d, s: (
        (d[f'cat_mu_class_{s}'] == 2)
        | (d[f'cat_class_star_{s}'] > 0.80)
    ),
    'JWST': lambda d, s: (
        d[f'cat_flag_star_{s}'].fillna(False).astype(bool)
    ),
    'Euclid_VIS': lambda d, s: (
        (d[f'cat_phz_classification_{s}'] == 1)
        | (d[f'cat_point_like_prob_{s}'] > 0.5)
    ),
    'Euclid_NISP': lambda d, s: (
        (d[f'cat_phz_classification_{s}'] == 1)
        | (d[f'cat_point_like_prob_{s}'] > 0.5)
    ),
}

MAG_BINS = np.arange(15.0, 28.5, 1.0)   # bins of 1 mag from 15 to 28


# ─────────────────────────────────────────────────────────────────────────────
# Phase A — load training set & compute per-mag thresholds
# ─────────────────────────────────────────────────────────────────────────────

def load_cw_flags() -> pd.DataFrame:
    """Load COSMOS-Web star/QSO/AGN flags keyed by jwst_id (= cw 'id')."""
    print('[setup] loading COSMOS-Web flags ...')
    with fits.open(CW_PATH) as hdul:
        photo = hdul['PHOTOMETRY HOTCOLD AND SE++'].data
        lephare = hdul['LEPHARE'].data
        cw_id        = np.asarray(photo['id']).astype(np.int64)
        flag_star    = np.asarray(photo['flag_star']).astype(bool)
        flag_star_h  = np.asarray(photo['flag_star_hsc']).astype(bool)
        lp_type      = np.asarray(lephare['type']).astype(np.int64)
        flag_chandra = np.asarray(lephare['flag_chandra']).astype(np.float64) > 0.5
    return pd.DataFrame({
        'cw_id':           cw_id,
        'cw_flag_star':    flag_star,
        'cw_flag_star_hsc': flag_star_h,
        'cw_lp_type':      lp_type,
        'cw_flag_chandra': flag_chandra,
    })


def build_training_set(cw_flags: pd.DataFrame) -> pd.DataFrame:
    """Load 4-way master, derive ellipticity per mission, label training rows.

    Labels:
      gaia_star : Gaia DR3 strict (classprob>0.95, ruwe<1.4, non_single_star=0)
      qso       : COSMOS-Web LePhare type==2 OR flag_chandra (X-ray AGN)
      galaxy    : LePhare type==0 AND NOT any of star/qso flags
      unknown   : everything else (excluded from training)

    We DROP the loose `star_cw` class (LePhare type==1) from training
    because it's contaminated with extended sources that pulled the
    99th-pctl threshold too high in the first pass.
    """
    print('[setup] loading 4-way master ...')
    df = pd.read_parquet(OUT / 'cross_HST_JWST_Euclid_VIS_Euclid_NISP.parquet')
    print(f'[setup]   {len(df):,} unique stars in 4-way')

    # Cross-link CW flags via jwst_id
    df['_jwst_id_int'] = pd.to_numeric(df['jwst_id_jwst'], errors='coerce').astype('Int64')
    df = df.merge(cw_flags, left_on='_jwst_id_int', right_on='cw_id', how='left')

    # Derive ellipticity columns for HST and JWST
    if 'cat_a_image_as_hst' in df.columns and 'cat_b_image_as_hst' in df.columns:
        a = df['cat_a_image_as_hst'].values
        b = df['cat_b_image_as_hst'].values
        with np.errstate(invalid='ignore', divide='ignore'):
            df['ellip_hst'] = 1.0 - np.where(a > 0, b / a, np.nan)
    if 'cat_axratio_jwst' in df.columns:
        ax = df['cat_axratio_jwst'].values
        with np.errstate(invalid='ignore'):
            df['ellip_jwst'] = 1.0 - np.where(np.isfinite(ax), ax, np.nan)

    # Labels
    is_gaia_star = (
        (df['gaia_classprob_dsc_combmod_star'] > 0.95)
        & (df['gaia_ruwe'] < 1.4)
        & (df['gaia_non_single_star'] == 0)
    ).fillna(False)
    is_qso = (
        (df['cw_lp_type'] == 2) | (df['cw_flag_chandra'].fillna(False))
    )
    is_galaxy = (
        (df['cw_lp_type'] == 0)
        & (~is_gaia_star) & (~is_qso)
        & (~df['cw_flag_star'].fillna(False))
        & (df['cw_lp_type'] != 1)
        & (~df['cw_flag_chandra'].fillna(False))
    )

    label = pd.Series('unknown', index=df.index, dtype=object)
    label[is_galaxy]    = 'galaxy'
    label[is_qso]       = 'qso'
    label[is_gaia_star] = 'gaia_star'    # wins ties

    df['_label'] = label
    counts = label.value_counts()
    print('[setup]   training labels (strict):')
    for k, v in counts.items():
        print(f'              {k:<14} {v:,}')
    return df


def compute_thresholds(df: pd.DataFrame) -> dict:
    """Per mission and per mag bin: 95th-pctl of ellipticity among
    UNSATURATED (mag > SATURATION_MAG[m]) gaia_star+qso positives.

    Returns:
      thr[mission] = {
        '_mag_col': str, '_ellip_col': str,
        '_sat_mag': float, '_psf_max_thr': float (cap),
        'ellip': np.array of per-mag-bin upper limits
      }
    """
    print('[setup] computing per-mag ellipticity thresholds (95th pctl) ...')
    positives = df['_label'].isin(['gaia_star', 'qso'])
    df_pos = df[positives]
    df_neg = df[df['_label'] == 'galaxy']
    print(f'[setup]   positives: {len(df_pos):,}, negatives: {len(df_neg):,}')

    thr = {}
    for mission in MORPH_ELLIP_FEATURE:
        sfx = TAG[mission]
        mag_col = f'{MAG_COL[mission]}_{sfx}'
        feat = MORPH_ELLIP_FEATURE[mission]
        ellip_col = feat if feat in ('ellip_hst', 'ellip_jwst') else f'{feat}_{sfx}'
        if feat == 'ellip':
            ellip_col = f'ellip_{sfx}'
        if mag_col not in df_pos.columns or ellip_col not in df_pos.columns:
            print(f'[setup]   {mission}: skipping (missing {mag_col} or {ellip_col})')
            continue
        sat_mag = SATURATION_MAG[mission]
        mag_p = df_pos[mag_col].values
        e_p   = df_pos[ellip_col].values
        # only use unsaturated positives
        unsat = mag_p > sat_mag
        per_bin = []
        for i in range(len(MAG_BINS) - 1):
            lo, hi = MAG_BINS[i], MAG_BINS[i+1]
            mp = unsat & (mag_p >= lo) & (mag_p < hi) & np.isfinite(e_p)
            if mp.sum() >= 10:
                thr_val = float(np.nanpercentile(e_p[mp], 95))
            else:
                thr_val = np.nan
            per_bin.append(thr_val)
        per_bin = np.array(per_bin)
        # Cap at a physical maximum (point sources should not have ellip > 0.30)
        per_bin = np.minimum(per_bin, 0.30)
        # Fill NaN bins with the median of non-NaN bins
        nan_mask = ~np.isfinite(per_bin)
        if nan_mask.any() and np.isfinite(per_bin).any():
            per_bin[nan_mask] = np.nanmedian(per_bin)
        thr[mission] = {
            '_mag_col': mag_col,
            '_ellip_col': ellip_col,
            '_sat_mag': sat_mag,
            'ellip': per_bin,
        }
        # Also compute false-positive rate on galaxies for sanity
        mag_g = df_neg[mag_col].values
        e_g   = df_neg[ellip_col].values
        bin_idx_g = np.digitize(mag_g, MAG_BINS) - 1
        bin_idx_g = np.clip(bin_idx_g, 0, len(MAG_BINS) - 2)
        with np.errstate(invalid='ignore'):
            galaxy_passes = np.isfinite(e_g) & (e_g < per_bin[bin_idx_g])
        fp_rate = galaxy_passes.sum() / max(1, np.isfinite(e_g).sum())
        # Recovery rate on positives
        bin_idx_p = np.digitize(mag_p, MAG_BINS) - 1
        bin_idx_p = np.clip(bin_idx_p, 0, len(MAG_BINS) - 2)
        with np.errstate(invalid='ignore'):
            pos_passes = unsat & np.isfinite(e_p) & (e_p < per_bin[bin_idx_p])
        tp_rate = pos_passes.sum() / max(1, (unsat & np.isfinite(e_p)).sum())
        print(f'[setup]   {mission:<11} ellip 95th-pctl bins '
              f'(mag>{sat_mag}): {per_bin.min():.3f}→{per_bin.max():.3f}, '
              f'galaxy FP rate={fp_rate*100:.1f}%, point-source TP rate={tp_rate*100:.1f}%')
    return thr


def save_threshold_table(thr: dict, path: Path) -> None:
    """Dump thresholds as a human-readable text file."""
    lines = ['Per-mag-bin ellipticity thresholds (Step 6, 95th-pctl of '
             'gaia_star+qso among unsaturated)\n']
    centres = 0.5 * (MAG_BINS[:-1] + MAG_BINS[1:])
    lines.append(f'Mag bin centres: ' + ' '.join(f'{c:6.2f}' for c in centres))
    for m, mdict in thr.items():
        lines.append(f'\n=== {m} (mag {mdict["_mag_col"]}, '
                     f'ellip {mdict["_ellip_col"]}, sat<{mdict["_sat_mag"]}) ===')
        vals = ' '.join(f'{v:6.3f}' if np.isfinite(v) else '   nan'
                        for v in mdict['ellip'])
        lines.append(f'  ellip 95%ile: [{vals}]')
    path.write_text('\n'.join(lines) + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — apply thresholds (parallel worker)
# ─────────────────────────────────────────────────────────────────────────────

def apply_pointsource_flags(combo_name: str, missions: list[str],
                            thresholds: dict, cw_flags: pd.DataFrame) -> dict:
    """Worker: load one cross-matched catalog, classify each row.

    Mission verdict (per-mission is_point_source_<m>):
      If detected_in_<m>:
        if mag <= SATURATION_MAG[m]:
          PS iff PRIMARY_PS_FLAG[m] is True (morphology unreliable)
        else (unsaturated):
          PS iff PRIMARY_PS_FLAG[m] is True AND ellip < per-mag-bin threshold

    Master is_point_source:
      ANY of:
        - is_point_source_<m> for any m in {detected missions}
        - gaia_anchored star (Gaia DR3 classprob_star>0.9 AND ruwe<1.4)
        - tie_type == 'gaia_only_bright'
        - COSMOS-Web QSO (lp_type==2) OR Chandra X-ray AGN
    """
    t0 = time.time()
    tag = '+'.join(TAG[m] for m in missions)
    print(f'[{tag}] starting ...')

    df = pd.read_parquet(OUT / f'cross_{combo_name}.parquet')
    n_total = len(df)
    print(f'[{tag}]   loaded {n_total:,} rows × {len(df.columns)} cols')

    # Cross-link CW flags via jwst_id (only if combo includes JWST)
    if 'jwst_id_jwst' in df.columns:
        df['_jwst_id_int'] = pd.to_numeric(df['jwst_id_jwst'], errors='coerce').astype('Int64')
        df = df.merge(cw_flags, left_on='_jwst_id_int', right_on='cw_id', how='left')

    # Derive ellipticity columns if needed
    if 'cat_a_image_as_hst' in df.columns and 'cat_b_image_as_hst' in df.columns:
        a = df['cat_a_image_as_hst'].values
        b = df['cat_b_image_as_hst'].values
        with np.errstate(invalid='ignore', divide='ignore'):
            df['ellip_hst'] = 1.0 - np.where(a > 0, b / a, np.nan)
    if 'cat_axratio_jwst' in df.columns:
        ax = df['cat_axratio_jwst'].values
        with np.errstate(invalid='ignore'):
            df['ellip_jwst'] = 1.0 - np.where(np.isfinite(ax), ax, np.nan)

    evidence_lists = [list() for _ in range(n_total)]
    is_ps_master  = np.zeros(n_total, dtype=bool)

    for m in missions:
        sfx = TAG[m]
        det_col = f'detected_in_{sfx}'
        if det_col not in df.columns:
            continue
        detected = df[det_col].values.astype(bool)
        ps_m = np.zeros(n_total, dtype=bool)

        # Primary catalog flag
        try:
            primary = PRIMARY_PS_FLAG[m](df, sfx).fillna(False).astype(bool).values
        except Exception:
            primary = np.zeros(n_total, dtype=bool)

        # Mag and ellipticity
        thr_m = thresholds.get(m, None)
        if thr_m is None:
            df[f'is_point_source_{sfx}'] = detected & primary
            ps_m = detected & primary
        else:
            mag_col = thr_m['_mag_col']
            ellip_col = thr_m['_ellip_col']
            sat_mag = thr_m['_sat_mag']
            ellip_arr = thr_m['ellip']
            mag = df[mag_col].values if mag_col in df.columns else np.full(n_total, np.nan)
            ellip = df[ellip_col].values if ellip_col in df.columns else np.full(n_total, np.nan)
            finite_mag = np.isfinite(mag)
            bin_idx = np.digitize(mag, MAG_BINS) - 1
            bin_idx = np.clip(bin_idx, 0, len(MAG_BINS) - 2)
            thr_for_row = np.where(finite_mag, ellip_arr[bin_idx], np.nan)
            saturated_bright = finite_mag & (mag <= sat_mag)
            unsaturated      = finite_mag & (mag >  sat_mag)
            with np.errstate(invalid='ignore'):
                ellip_pass = np.isfinite(ellip) & (ellip < thr_for_row)

            # Saturated mag range: trust catalog flag only
            ps_m_sat   = saturated_bright & primary & detected
            # Unsaturated: require both catalog flag AND ellipticity cut
            ps_m_unsat = unsaturated & primary & ellip_pass & detected
            ps_m = ps_m_sat | ps_m_unsat

            for i in np.where(ps_m_sat)[0]:
                evidence_lists[i].append(f'{sfx}:saturated+flag')
            for i in np.where(ps_m_unsat)[0]:
                evidence_lists[i].append(f'{sfx}:flag+ellip<thr')

        df[f'is_point_source_{sfx}'] = ps_m
        is_ps_master |= ps_m

    # ── Gaia / gaia_only_bright / QSO override ─────────────────────────────
    if 'gaia_classprob_dsc_combmod_star' in df.columns:
        is_gaia_pt = (
            (df['gaia_classprob_dsc_combmod_star'] > 0.9)
            & (df['gaia_ruwe'] < 1.4)
        ).fillna(False).values
        is_ps_master |= is_gaia_pt
        for i in np.where(is_gaia_pt)[0]:
            evidence_lists[i].append('gaia:classprob>0.9_ruwe<1.4')

    if 'tie_type' in df.columns:
        gob = (df['tie_type'].values == 'gaia_only_bright')
        is_ps_master |= gob
        for i in np.where(gob)[0]:
            evidence_lists[i].append('gaia_only_bright')

    # COSMOS-Web QSO/AGN override (when JWST is in the combo)
    if 'cw_lp_type' in df.columns:
        is_qso = (
            (df['cw_lp_type'] == 2)
            | df['cw_flag_chandra'].fillna(False).astype(bool)
        ).values
        is_ps_master |= is_qso
        for i in np.where(is_qso)[0]:
            evidence_lists[i].append('cw:qso/agn')

    df['is_point_source'] = is_ps_master
    df['point_source_evidence'] = [','.join(ev) for ev in evidence_lists]

    # Drop the merge helper columns before writing
    for c in ('_jwst_id_int', 'cw_id', 'cw_flag_star', 'cw_flag_star_hsc',
              'cw_lp_type', 'cw_flag_chandra', 'ellip_hst', 'ellip_jwst'):
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Write
    out_parq = OUT / f'refined_{combo_name}.parquet'
    df.to_parquet(out_parq, index=False)
    csv_written = False
    if len(df) * len(df.columns) < 200_000_000:
        df.to_csv(OUT / f'refined_{combo_name}.csv', index=False)
        csv_written = True

    n_ps = int(is_ps_master.sum())
    elapsed = time.time() - t0
    print(f'[{tag}]   wrote {out_parq.name}: {n_total:,} rows '
          f'({n_ps:,} flagged is_point_source = {100*n_ps/n_total:.2f}%) in {elapsed:.1f}s')

    return {
        'combo': combo_name,
        'missions': missions,
        'n_total': n_total,
        'n_point_source': n_ps,
        'frac_point_source': n_ps / n_total,
        'elapsed_s': elapsed,
        'csv_written': csv_written,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots: per-mission morphology vs mag for positives vs negatives
# ─────────────────────────────────────────────────────────────────────────────

def make_diagnostic_plots(df: pd.DataFrame, thr: dict) -> None:
    print('[setup] making diagnostic plots ...')
    for m, mdict in thr.items():
        sfx = TAG[m]
        mag_col = mdict['_mag_col']
        ellip_col = mdict['_ellip_col']
        if mag_col not in df.columns or ellip_col not in df.columns:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        for label, color, size in [('galaxy', 'C7', 1), ('gaia_star', 'C0', 4),
                                    ('qso', 'C3', 4)]:
            sub = df[df['_label'] == label]
            if len(sub) == 0:
                continue
            ax.scatter(sub[mag_col], sub[ellip_col],
                       s=size, alpha=0.5, color=color,
                       rasterized=True, label=f'{label} (N={len(sub):,})')
        xs = 0.5 * (MAG_BINS[:-1] + MAG_BINS[1:])
        ax.plot(xs, mdict['ellip'], 'k-', lw=2, label='95% ellip thr')
        ax.axvline(mdict['_sat_mag'], color='r', ls='--', lw=1,
                   label=f'sat={mdict["_sat_mag"]}')
        ax.set_xlabel(mag_col.replace('_', ' '))
        ax.set_ylabel(ellip_col)
        ax.set_title(f'{m}: ellipticity vs magnitude (training)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = HTML / f'pointsource_thresholds_{m}.png'
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f'[setup]   {out_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

COMBINATIONS = [
    (['HST', 'JWST'],                                          'HST_JWST'),
    (['HST', 'Euclid_VIS'],                                    'HST_Euclid_VIS'),
    (['HST', 'Euclid_NISP'],                                   'HST_Euclid_NISP'),
    (['JWST', 'Euclid_VIS'],                                   'JWST_Euclid_VIS'),
    (['JWST', 'Euclid_NISP'],                                  'JWST_Euclid_NISP'),
    (['Euclid_VIS', 'Euclid_NISP'],                            'Euclid_VIS_Euclid_NISP'),
    (['HST', 'JWST', 'Euclid_VIS'],                            'HST_JWST_Euclid_VIS'),
    (['HST', 'JWST', 'Euclid_NISP'],                           'HST_JWST_Euclid_NISP'),
    (['HST', 'JWST', 'Euclid_VIS', 'Euclid_NISP'],
     'HST_JWST_Euclid_VIS_Euclid_NISP'),
]


def main():
    t_total = time.time()
    print('=' * 78)
    print('Step 6 — refine point-source catalog (per-mag-bin transparent cuts)')
    print('=' * 78)

    # Phase A — setup (serial)
    cw_flags = load_cw_flags()
    train_df = build_training_set(cw_flags)
    thresholds = compute_thresholds(train_df)
    save_threshold_table(thresholds, HTML / 'pointsource_threshold_table.txt')
    make_diagnostic_plots(train_df, thresholds)
    print()

    # Phase B — apply in parallel
    print('=' * 78)
    print('Applying thresholds to 9 catalogs in parallel ...')
    print('=' * 78)
    results = []
    with ProcessPoolExecutor(max_workers=9) as pool:
        futures = {pool.submit(apply_pointsource_flags, name, miss, thresholds, cw_flags): name
                   for miss, name in COMBINATIONS}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f'[{name}] FAILED: {e!r}')
                import traceback
                traceback.print_exc()
                raise

    # Summary
    print()
    print('=' * 78)
    print('Summary')
    print('=' * 78)
    lines = []
    lines.append(f'{"output":<40} {"n_total":>10} {"is_point_source":>17} '
                 f'{"frac":>8} {"time":>7}')
    lines.append('-' * 88)
    for r in sorted(results, key=lambda x: (len(x['missions']), x['combo'])):
        lines.append(
            f'refined_{r["combo"]:<32} {r["n_total"]:>10,} '
            f'{r["n_point_source"]:>17,} {r["frac_point_source"]*100:>7.2f}% '
            f'{r["elapsed_s"]:>6.1f}s'
        )
    text = '\n'.join(lines)
    print(text)
    (HTML / 'refined_pointsource_summary.txt').write_text(text + '\n')
    print(f'\nTotal wall time: {time.time() - t_total:.1f}s')


if __name__ == '__main__':
    main()
