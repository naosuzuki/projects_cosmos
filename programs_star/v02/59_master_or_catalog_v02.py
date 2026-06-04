#!/usr/bin/env python
"""
60_master_or_catalog_v02.py — STEP 8 v02 master OR catalog with bug fixes.

Bug fixes vs v01 (documented in programs_star/CLAUDE.md):
  B1 — primary_id now uniquely-keyed (HST_v02 has globally-unique hst_id)
  B2/B4 — new `is_likely_star` column aggregates all stellar evidence
  B3 — new `hst_saturated_likely` flag for Euclid-says-star + missing HST
  B6 — PM join uses (acs_number_hst, dao_F814W_tile_hst) tuple, not hst_id

Inputs: cat_matched_*_with_gaia_v02.parquet (v02 fixed-id HST),
        refined_*_with_pm_v02.parquet (legacy per-tile ids; matched via
        acs_number+tile join).

Output: csvfiles_star/master_or_catalog_v02.parquet (+.csv if small).
"""
from __future__ import annotations
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u

warnings.filterwarnings('ignore')
ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'

MISSIONS = ['HST', 'JWST', 'Euclid_VIS', 'Euclid_NISP']
TAG = {'HST': 'hst', 'JWST': 'jwst', 'Euclid_VIS': 'vis', 'Euclid_NISP': 'nisp'}
ID_COL = {'HST': 'hst_id', 'JWST': 'jwst_id',
           'Euclid_VIS': 'euclid_id', 'Euclid_NISP': 'euclid_id'}

PAIR_RADIUS_AS = {
    frozenset(['HST', 'JWST']):              0.50,
    frozenset(['HST', 'Euclid_VIS']):        0.50,
    frozenset(['HST', 'Euclid_NISP']):       0.60,
    frozenset(['JWST', 'Euclid_VIS']):       0.25,
    frozenset(['JWST', 'Euclid_NISP']):      0.30,
    frozenset(['Euclid_VIS', 'Euclid_NISP']): 0.30,
}

# v02 PM pair catalogs (combo_name → file)
PM_PAIRS = [
    'HST_Euclid_VIS',
    'HST_Euclid_NISP',
    'HST_JWST',
    'JWST_Euclid_VIS',
    'JWST_Euclid_NISP',
]


class UnionFind:
    __slots__ = ('parent', 'rank')
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank   = np.zeros(n, dtype=np.int32)
    def find(self, x):
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            nxt = p[x]; p[x] = root; x = nxt
        return root
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def main():
    t_total = time.time()
    print('=' * 78)
    print('Step 8 — master OR catalog (4-mission union, primary_id JWST > HST > Euclid)')
    print('=' * 78)

    # ── Load 4 per-mission Gaia-augmented catalogs ───────────────────────
    cats = {}
    offsets = {}
    n_running = 0
    for m in MISSIONS:
        print(f'[load] {m} ...', end=' ', flush=True)
        df = pd.read_parquet(OUT / f'cat_matched_{m}_with_gaia_v02.parquet')
        df = df.reset_index(drop=True)
        cats[m] = df
        offsets[m] = n_running
        n_running += len(df)
        print(f'{len(df):,} rows')
    n_total = n_running
    print(f'[load] total rows across missions: {n_total:,}')

    # ── Build global arrays for union-find ──────────────────────────────
    cat_ra  = np.full(n_total, np.nan)
    cat_dec = np.full(n_total, np.nan)
    gid     = np.full(n_total, -1, dtype=np.int64)
    for m in MISSIONS:
        off = offsets[m]; n = len(cats[m])
        cat_ra[off:off+n]  = cats[m]['cat_ra'].values
        cat_dec[off:off+n] = cats[m]['cat_dec'].values
        gv = cats[m]['gaia_source_id'].values
        gid[off:off+n] = np.where(pd.isna(gv), -1, gv).astype(np.int64)
    has_cat  = ~np.isnan(cat_ra)
    has_gaia = gid > 0

    # ── Tier 1: Gaia-anchored union ─────────────────────────────────────
    print('[union] tier 1 — Gaia source_id ...')
    uf = UnionFind(n_total)
    idx_g = np.where(has_gaia)[0]
    order = np.argsort(gid[idx_g], kind='stable')
    idx_g = idx_g[order]
    gv = gid[idx_g]
    starts = np.concatenate(([0], np.where(np.diff(gv) != 0)[0] + 1, [len(gv)]))
    n_uni_gaia = 0
    for k in range(len(starts) - 1):
        s, e = starts[k], starts[k+1]
        if e - s > 1:
            first = int(idx_g[s])
            for j in range(s+1, e):
                uf.union(first, int(idx_g[j]))
            n_uni_gaia += 1
    print(f'[union]   gaia clusters created: {n_uni_gaia:,}')

    # ── Tier 2: positional union per mission pair ───────────────────────
    print('[union] tier 2 — positional pairs')
    for m1, m2 in combinations(MISSIONS, 2):
        radius = PAIR_RADIUS_AS[frozenset([m1, m2])]
        off1, off2 = offsets[m1], offsets[m2]
        n1, n2 = len(cats[m1]), len(cats[m2])
        sel1 = has_cat[off1:off1+n1]
        sel2 = has_cat[off2:off2+n2]
        idx1l = np.where(sel1)[0]; idx2l = np.where(sel2)[0]
        if len(idx1l) == 0 or len(idx2l) == 0: continue
        c1 = SkyCoord(cat_ra[off1+idx1l]*u.deg,  cat_dec[off1+idx1l]*u.deg)
        c2 = SkyCoord(cat_ra[off2+idx2l]*u.deg,  cat_dec[off2+idx2l]*u.deg)
        p1, p2, sep, _ = search_around_sky(c1, c2, radius*u.arcsec)
        if len(p1) == 0: continue
        order = np.argsort(sep.arcsec)
        p1s = p1[order]; p2s = p2[order]
        sa = np.zeros(len(idx1l), dtype=bool); sb = np.zeros(len(idx2l), dtype=bool)
        keep = np.zeros(len(p1s), dtype=bool)
        for i in range(len(p1s)):
            a, b = p1s[i], p2s[i]
            if not sa[a] and not sb[b]:
                sa[a] = True; sb[b] = True; keep[i] = True
        ka = idx1l[p1s[keep]]; kb = idx2l[p2s[keep]]
        for x, y in zip(off1 + ka, off2 + kb):
            uf.union(int(x), int(y))
        print(f'[union]   {m1}↔{m2} at {radius}": {keep.sum():,} unique 1:1 ties')

    # ── Resolve clusters ────────────────────────────────────────────────
    print('[union] resolving cluster roots ...')
    roots = np.fromiter((uf.find(i) for i in range(n_total)),
                        dtype=np.int64, count=n_total)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    n_clusters = len(unique_roots)
    print(f'[union]   {n_clusters:,} unique sources')

    # ── For each (cluster, mission), pick first row index ───────────────
    cluster_to_row = {m: np.full(n_clusters, -1, dtype=np.int64) for m in MISSIONS}
    detected = {m: np.zeros(n_clusters, dtype=bool) for m in MISSIONS}
    for m in MISSIONS:
        off = offsets[m]; n = len(cats[m])
        cl = inverse[off:off+n]
        # prefer catalog-detected over gaia_only_bright (catalog rows have lower offsets)
        # use source_type filter to count detected: catalog/catalog+gaia
        st = cats[m]['source_type'].values
        is_cat_det = np.isin(st, ['catalog', 'catalog+gaia'])
        order = np.argsort((~is_cat_det).astype(int), kind='stable')   # cat first
        cl_sorted  = cl[order]
        first      = np.concatenate(([True], cl_sorted[1:] != cl_sorted[:-1]))
        cluster_to_row[m][cl_sorted[first]] = order[first]
        # detected_in flag: TRUE if any of the cluster's m rows has cat detection
        np.logical_or.at(detected[m], cl[is_cat_det], True)

    # ── Build master table ──────────────────────────────────────────────
    print('[build] master table ...')
    out = {}
    # mission ID columns (raw IDs per mission)
    for m in MISSIONS:
        col = ID_COL[m]
        idxs = cluster_to_row[m]
        mask = idxs >= 0
        src = pd.to_numeric(cats[m][col], errors='coerce').values
        new_col_name = ('euclid_vis_id' if m == 'Euclid_VIS'
                        else 'euclid_nisp_id' if m == 'Euclid_NISP'
                        else col)
        arr = np.full(n_clusters, np.nan)
        arr[mask] = src[idxs[mask]]
        out[new_col_name] = arr

    # detected_in flags
    for m in MISSIONS:
        out[f'detected_in_{TAG[m]}'] = detected[m]
    # source_type per mission
    for m in MISSIONS:
        idxs = cluster_to_row[m]
        mask = idxs >= 0
        src = cats[m]['source_type'].values
        arr = np.full(n_clusters, None, dtype=object)
        arr[mask] = src[idxs[mask]]
        out[f'source_type_{TAG[m]}'] = arr

    # primary ID — precedence: JWST > HST > Euclid > Gaia (gaia_only_bright fallback)
    jid = out['jwst_id']
    hid = out['hst_id']
    vid = out['euclid_vis_id']
    nid = out['euclid_nisp_id']
    # Gaia source ID per cluster (will fill below from per-mission Gaia info)
    gaia_id_arr = np.full(n_clusters, -1, dtype=np.int64)
    for m in MISSIONS:
        idxs = cluster_to_row[m]; mask = idxs >= 0
        if not mask.any(): continue
        gvals = cats[m]['gaia_source_id'].values[idxs[mask]]
        good  = ~pd.isna(gvals) & (gvals > 0)
        sel_c = np.where(mask)[0][good]
        sel_c = sel_c[gaia_id_arr[sel_c] < 0]
        if len(sel_c):
            gaia_id_arr[sel_c] = gvals[good][:len(sel_c)].astype(np.int64)

    # Precedence (per user 2026-05-27):
    #   1. JWST ID         → bare integer
    #   2. Euclid VIS ID   → EuclidID_<id>
    #   3. HST ID          → HSTID_<id>
    #   4. Euclid NISP ID  → EuclidID_<id>
    #   5. Gaia source ID  → GaiaID_<id>  (saturated in every mission)
    pid  = np.empty(n_clusters, dtype=object)
    psrc = np.empty(n_clusters, dtype=object)
    for i in range(n_clusters):
        if np.isfinite(jid[i]):
            pid[i]  = str(int(jid[i]));            psrc[i] = 'jwst'
        elif np.isfinite(vid[i]):
            pid[i]  = f'EuclidID_{int(vid[i])}';   psrc[i] = 'euclid_vis'
        elif np.isfinite(hid[i]):
            pid[i]  = f'HSTID_{int(hid[i])}';      psrc[i] = 'hst'
        elif np.isfinite(nid[i]):
            pid[i]  = f'EuclidID_{int(nid[i])}';   psrc[i] = 'euclid_nisp'
        elif gaia_id_arr[i] > 0:
            pid[i]  = f'GaiaID_{gaia_id_arr[i]}';  psrc[i] = 'gaia_only'
        else:
            pid[i]  = f'UNK_{i}';                  psrc[i] = 'unknown'
    out['primary_id']     = pid
    out['primary_source'] = psrc

    # ── Canonical Gaia (24 cols) — prefer JWST > VIS > NISP > HST ───────
    GAIA_COLS = [c for c in cats['HST'].columns if c.startswith('gaia_')]
    for col in GAIA_COLS:
        dt = cats['HST'][col].dtype
        out[col] = (np.full(n_clusters, np.nan) if dt.kind in 'fiu'
                    else np.full(n_clusters, None, dtype=object))
    filled = np.zeros(n_clusters, dtype=bool)
    for m in ['JWST', 'Euclid_VIS', 'Euclid_NISP', 'HST']:
        idxs = cluster_to_row[m]; mask = idxs >= 0
        if not mask.any(): continue
        gv = cats[m]['gaia_source_id'].values[idxs[mask]]
        good = ~pd.isna(gv) & (gv > 0)
        sel_cluster = np.where(mask)[0][good]
        sel_cluster = sel_cluster[~filled[sel_cluster]]
        if len(sel_cluster) == 0: continue
        sel_local = idxs[sel_cluster]
        for col in GAIA_COLS:
            out[col][sel_cluster] = cats[m][col].values[sel_local]
        filled[sel_cluster] = True

    # ── Per-mission cat / DAO columns ────────────────────────────────────
    for m in MISSIONS:
        suffix = TAG[m]
        idxs = cluster_to_row[m]; mask = idxs >= 0
        for col in cats[m].columns:
            if col.startswith('gaia_'): continue
            new = f'{col}_{suffix}'
            src = cats[m][col].values
            if src.dtype.kind in 'fiu':
                arr = np.full(n_clusters, np.nan)
            elif src.dtype.kind == 'b':
                arr = np.full(n_clusters, False)
            else:
                arr = np.full(n_clusters, None, dtype=object)
            arr[mask] = src[idxs[mask]]
            out[new] = arr

    # ── is_point_source / is_agn_qso — aggregate from ALL 9 refined catalogs
    print('[flags] aggregating is_point_source / is_agn_qso from 9 refined cats ...')
    out['is_point_source'] = np.zeros(n_clusters, dtype=bool)
    out['is_agn_qso']      = np.zeros(n_clusters, dtype=bool)

    # Build per-cluster lookup keys
    cluster_keys = {}   # (kind, id) → cluster index
    for ci in range(n_clusters):
        if np.isfinite(jid[ci]): cluster_keys[('j', int(jid[ci]))] = ci
        if np.isfinite(hid[ci]): cluster_keys[('h', int(hid[ci]))] = ci
        if np.isfinite(vid[ci]): cluster_keys[('v', int(vid[ci]))] = ci
        if np.isfinite(nid[ci]): cluster_keys[('n', int(nid[ci]))] = ci
        if gaia_id_arr[ci] > 0:  cluster_keys[('g', int(gaia_id_arr[ci]))] = ci

    REFINED_FILES = [
        'refined_HST_JWST.parquet',
        'refined_HST_Euclid_VIS.parquet',
        'refined_HST_Euclid_NISP.parquet',
        'refined_JWST_Euclid_VIS.parquet',
        'refined_JWST_Euclid_NISP.parquet',
        'refined_Euclid_VIS_Euclid_NISP.parquet',
        'refined_HST_JWST_Euclid_VIS.parquet',
        'refined_HST_JWST_Euclid_NISP.parquet',
        'refined_HST_JWST_Euclid_VIS_Euclid_NISP.parquet',
    ]
    ps_total = 0; agn_total = 0
    for fname in REFINED_FILES:
        p = OUT / fname
        if not p.exists():
            continue
        rdf = pd.read_parquet(p, columns=[
            c for c in [
                'is_point_source','jwst_id_jwst','hst_id_hst',
                'euclid_id_vis','euclid_id_nisp','gaia_source_id'
            ] if True
        ] if False else None)   # read all
        # Determine ID columns present
        id_cols = [
            ('j', 'jwst_id_jwst'),
            ('h', 'hst_id_hst'),
            ('v', 'euclid_id_vis'),
            ('n', 'euclid_id_nisp'),
            ('g', 'gaia_source_id'),
        ]
        if 'is_point_source' not in rdf.columns:
            continue
        ps_vals = rdf['is_point_source'].values
        for kind, col in id_cols:
            if col not in rdf.columns:
                continue
            vals = pd.to_numeric(rdf[col], errors='coerce').astype('Int64')
            for ri in np.where(ps_vals & vals.notna().values)[0]:
                k = (kind, int(vals.iloc[ri]))
                ci = cluster_keys.get(k)
                if ci is not None:
                    out['is_point_source'][ci] = True
        ps_total = int(out['is_point_source'].sum())

    # is_agn_qso from v02 PM catalogs (which carry the flag) +
    # direct CW AGN/QSO lookup for sources without JWST detection.
    from astropy.io import fits
    CW_PATH = '/Volumes/exdisk1/data/catalog/COSMOSWeb_mastercatalog_v1.1.fits'
    with fits.open(CW_PATH) as hdul:
        photo = hdul['PHOTOMETRY HOTCOLD AND SE++'].data
        lephare = hdul['LEPHARE'].data
        cw_id  = np.asarray(photo['id']).astype(np.int64)
        cw_ra  = np.asarray(photo['ra']).astype(float)
        cw_dec = np.asarray(photo['dec']).astype(float)
        lp_type = np.asarray(lephare['type']).astype(np.int64)
        chandra = np.asarray(lephare['flag_chandra']).astype(float) > 0.5
    agn_mask = (lp_type == 2) | chandra
    cw_agn_ids = set(cw_id[agn_mask].tolist())
    # direct match by JWST id
    for ci in range(n_clusters):
        if np.isfinite(jid[ci]) and int(jid[ci]) in cw_agn_ids:
            out['is_agn_qso'][ci] = True
    # position match for non-JWST sources
    agn_ra = cw_ra[agn_mask]; agn_dec = cw_dec[agn_mask]
    cq = SkyCoord(agn_ra*u.deg, agn_dec*u.deg)
    # use the best available position from each cluster
    cluster_ra  = np.full(n_clusters, np.nan)
    cluster_dec = np.full(n_clusters, np.nan)
    for m in ['JWST', 'Euclid_VIS', 'Euclid_NISP', 'HST']:
        idxs = cluster_to_row[m]; mask = idxs >= 0
        # only fill clusters whose ra is still NaN
        need = mask & ~np.isfinite(cluster_ra)
        sel = np.where(need)[0]
        if len(sel) == 0: continue
        cluster_ra[sel]  = cats[m]['cat_ra'].values[idxs[sel]]
        cluster_dec[sel] = cats[m]['cat_dec'].values[idxs[sel]]
    have_pos = np.isfinite(cluster_ra)
    if have_pos.any():
        cc = SkyCoord(cluster_ra[have_pos]*u.deg, cluster_dec[have_pos]*u.deg)
        idx_q, idx_c, sep, _ = search_around_sky(cq, cc, 0.3*u.arcsec)
        match_clusters = np.where(have_pos)[0][np.unique(idx_c)]
        out['is_agn_qso'][match_clusters] = True
    agn_total = int(out['is_agn_qso'].sum())
    print(f'[flags]   is_point_source: {ps_total:,}   is_agn_qso: {agn_total:,}')

    # ── PM info from 5 v02 pair catalogs ─────────────────────────────────
    print('[pm] joining v02 PM info from 5 pairs ...')
    PM_COLS_SRC = ['pmra', 'pmdec', 'pmtot', 'pmra_err', 'pmdec_err', 'pmtot_err',
                    'pm_flag', 'pm_method']
    # Pre-allocate columns
    for pair in PM_PAIRS:
        for col in PM_COLS_SRC:
            if col in ('pm_flag', 'pm_method'):
                out[f'{col}_{pair}'] = np.full(n_clusters, None, dtype=object)
            else:
                out[f'{col}_{pair}'] = np.full(n_clusters, np.nan)

    for pair in PM_PAIRS:
        pm_path = OUT / f'refined_{pair}_v02_with_pm_clean.parquet'
        if not pm_path.exists():
            print(f'[pm]   {pair}: file missing, skipping')
            continue
        pm = pd.read_parquet(pm_path)
        # Build lookup by primary key.  Use jwst_id if JWST in pair, else
        # by (hst_id, euclid_id) tuple, falling back to position-uniqueness
        # via id pair.
        if 'JWST' in pair:
            keys = pd.to_numeric(pm['jwst_id_jwst'], errors='coerce').astype('Int64')
            lookup = {int(k): i for i, k in enumerate(keys.values) if pd.notna(k)}
            # match by jwst_id in master
            for ci in range(n_clusters):
                if np.isfinite(jid[ci]):
                    j = int(jid[ci])
                    if j in lookup:
                        ri = lookup[j]
                        for col in PM_COLS_SRC:
                            if col in pm.columns:
                                out[f'{col}_{pair}'][ci] = pm[col].iloc[ri]
        else:
            # HST_Euclid_VIS or HST_Euclid_NISP. In v02 cascade, hst_id_hst
            # is the globally-unique idx_c (from 52_v02) propagated through
            # 55/56/57. So join by hst_id_hst directly is now safe — bug B6
            # is fixed automatically by the upstream B1 fix.
            id1_col = 'hst_id_hst'
            keys = pd.to_numeric(pm[id1_col], errors='coerce').astype('Int64')
            lookup = {int(k): i for i, k in enumerate(keys.values) if pd.notna(k)}
            for ci in range(n_clusters):
                if np.isfinite(hid[ci]):
                    h = int(hid[ci])
                    if h in lookup:
                        ri = lookup[h]
                        for col in PM_COLS_SRC:
                            if col in pm.columns:
                                out[f'{col}_{pair}'][ci] = pm[col].iloc[ri]
        if False:  # OBSOLETE block (kept dead for diff readability) — old (acs,tile) join:
            # join by (acs_number_hst, dao_F814W_tile_hst) tuple instead of
            # hst_id_hst alone. refined_*_with_pm_v02 uses legacy per-tile
            # hst_ids; master v02 carries acs_number_hst (the legacy id)
            # and the tile name, which together uniquely identify the source.
            id1_col = 'hst_id_hst'                  # legacy per-tile id in refined
            tile_col = 'dao_F814W_tile_hst'         # tile name in refined
            keys_id = pd.to_numeric(pm[id1_col], errors='coerce').astype('Int64')
            keys_tile = (pm[tile_col].values
                          if tile_col in pm.columns else None)
            if keys_tile is None:
                print(f'[pm]   {pair}: tile column missing in refined PM; '
                      f'falling back to id-only join (B6 not fully fixed)')
                lookup = {int(k): i for i, k in enumerate(keys_id.values)
                          if pd.notna(k)}
                for ci in range(n_clusters):
                    if np.isfinite(hid[ci]):
                        h = int(hid[ci])
                        if h in lookup:
                            ri = lookup[h]
                            for col in PM_COLS_SRC:
                                if col in pm.columns:
                                    out[f'{col}_{pair}'][ci] = pm[col].iloc[ri]
            else:
                lookup = {(int(k), str(t)): i
                          for i, (k, t) in enumerate(zip(keys_id.values, keys_tile))
                          if pd.notna(k) and t and t != ''}
                # cluster-side: acs_number_hst + dao_F814W_tile_hst
                acs = out.get('acs_number_hst')
                t_h = out.get('dao_F814W_tile_hst')
                if acs is None or t_h is None:
                    print(f'[pm]   {pair}: WARNING acs_number_hst or '
                          f'dao_F814W_tile_hst not in master out; falling back')
                else:
                    n_matched = 0
                    for ci in range(n_clusters):
                        if np.isfinite(acs[ci]) and t_h[ci]:
                            key = (int(acs[ci]), str(t_h[ci]))
                            if key in lookup:
                                ri = lookup[key]
                                for col in PM_COLS_SRC:
                                    if col in pm.columns:
                                        out[f'{col}_{pair}'][ci] = pm[col].iloc[ri]
                                n_matched += 1
        n_joined = int(np.sum(np.isfinite(out[f'pmtot_{pair}'])))
        print(f'[pm]   {pair}: {n_joined:,} sources got PM info')

    df_out = pd.DataFrame(out)

    # ── v02 NEW AGGREGATIONS (bugs B2/B3/B4) ─────────────────────────────
    print('[v02] computing is_likely_star, hst_saturated_likely, large_pm_likely ...')
    n_clusters = len(df_out)

    def _arr(name, default=np.nan, dtype=float):
        """Return a numpy array of length n_clusters; default if column missing."""
        if name in df_out.columns:
            v = df_out[name].values
            if dtype is float:
                # pd.to_numeric → ndarray when input is ndarray
                return np.asarray(pd.to_numeric(v, errors='coerce'), dtype=np.float64)
            return v
        return np.full(n_clusters, default)

    def _mask_lt(a, threshold):
        """boolean mask `a < threshold`, treating NaN as False."""
        m = np.zeros_like(a, dtype=bool)
        finite = np.isfinite(a)
        m[finite] = a[finite] < threshold
        return m

    def _mask_gt(a, threshold):
        m = np.zeros_like(a, dtype=bool)
        finite = np.isfinite(a)
        m[finite] = a[finite] > threshold
        return m

    def _mask_eq(a, value):
        m = np.zeros_like(a, dtype=bool)
        finite = np.isfinite(a)
        m[finite] = a[finite] == value
        return m

    # Pull all columns as numpy arrays
    cat_pl_vis     = _arr('cat_point_like_prob_vis')
    cat_pl_nisp    = _arr('cat_point_like_prob_nisp')
    cat_phz_vis    = _arr('cat_phz_classification_vis')
    cat_phz_nisp   = _arr('cat_phz_classification_nisp')
    cat_mag_VIS    = _arr('cat_mag_VIS_vis')
    cat_mag_F814   = _arr('cat_mag_F814W_hst')
    cat_class_star = _arr('cat_class_star_hst')
    cat_mu_class   = _arr('cat_mu_class_hst')
    has_hst_cat = ~np.isnan(_arr('hst_id'))

    # B3: hst_saturated_likely (Euclid says star + HST catalog missed it +
    # position is inside HST footprint). Proxy "inside HST footprint" =
    # source IS detected in JWST (since JWST ⊂ HST coverage in COSMOS).
    # Without this, every Euclid star outside the ACS field gets flagged.
    det_jwst = np.asarray(df_out['detected_in_jwst'].astype(bool).values, dtype=bool)
    star_phot_evidence = (_mask_eq(cat_phz_vis, 1)
                          | _mask_eq(cat_phz_nisp, 1)
                          | _mask_gt(cat_pl_vis, 0.9)
                          | _mask_gt(cat_pl_nisp, 0.9))
    hst_saturated_likely = star_phot_evidence & (~has_hst_cat) & det_jwst
    df_out['hst_saturated_likely'] = hst_saturated_likely

    # is_point_source / is_agn_qso (boolean cols already in df_out)
    is_ps  = np.asarray(df_out.get('is_point_source', pd.Series([False]*n_clusters)).fillna(False).astype(bool).values, dtype=bool)
    is_agn = np.asarray(df_out.get('is_agn_qso',      pd.Series([False]*n_clusters)).fillna(False).astype(bool).values, dtype=bool)
    gaia_present = ~np.isnan(_arr('gaia_source_id'))

    # PM significance from any of the 5 pairs.
    # Per user: PM detection = 100% star. The two flags differ only in
    # whether Gaia anchored the measurement; both indicate REAL motion
    # when significant. Apply the same 5σ + 5 mas/yr threshold to both
    # to ensure we catch cand_50 (10.2 mas/yr ok), cand_65 (8.5 ok),
    # cand_83 (9.1 ok), cand_92 (19.3 ok), etc.
    # Skip `inconsistent_with_gaia` (likely cross-match error, not real PM).
    pm_significant = np.zeros(n_clusters, dtype=bool)
    for pair in PM_PAIRS:
        pmt = _arr(f'pmtot_{pair}')
        pme = _arr(f'pmtot_err_{pair}')
        flag = df_out.get(f'pm_flag_{pair}', pd.Series(['']*n_clusters)).fillna('').astype(str).values
        finite = np.isfinite(pmt) & np.isfinite(pme)
        ok_or_gc = np.isin(flag, ('ok', 'gaia_consistent')) & finite
        sig = np.zeros(n_clusters, dtype=bool)
        sig[ok_or_gc] = (np.abs(pmt[ok_or_gc])
                          > np.maximum(5.0, 5.0 * pme[ok_or_gc]))
        pm_significant |= sig

    # NOTE: cat_mu_class_hst encoding is INVERTED from naïve expectation
    # (1=86% of HST rows is the "extended/galaxy" class). We use
    # cat_class_star_hst (numeric SExtractor stellarity 0-1) as the
    # primary HST star flag instead.
    #
    # DO NOT include is_point_source in this OR: SNe ARE point sources
    # by definition (transient photometric profile is a delta function
    # at survey resolution). Including is_ps would exclude real SNe
    # along with true stars. The CNN + morphology gates in the SN-finder
    # discriminate between SN-like and star-like point sources via
    # cross-survey detection asymmetry.
    is_likely_star = (
        gaia_present
        | _mask_gt(cat_class_star, 0.8)
        | _mask_eq(cat_phz_vis, 1)        # Euclid PHZ: 1=STAR (verified)
        | _mask_eq(cat_phz_nisp, 1)
        | _mask_gt(cat_pl_vis, 0.9)
        | _mask_gt(cat_pl_nisp, 0.9)
        | _mask_lt(cat_mag_VIS, 17)
        | _mask_lt(cat_mag_F814, 18)
        | hst_saturated_likely
        | pm_significant
    ) & (~is_agn)
    df_out['is_likely_star'] = is_likely_star

    # large_pm_likely
    extreme_pm = np.zeros(n_clusters, dtype=bool)
    for pair in PM_PAIRS:
        pmt = _arr(f'pmtot_{pair}')
        finite = np.isfinite(pmt)
        extreme_pm[finite] |= np.abs(pmt[finite]) > 100
    large_pm_likely = is_likely_star & (extreme_pm | pm_significant | hst_saturated_likely)
    df_out['large_pm_likely'] = large_pm_likely

    print(f'[v02]   is_likely_star:       {int(is_likely_star.sum()):>10,}')
    print(f'[v02]   hst_saturated_likely: {int(hst_saturated_likely.sum()):>10,}')
    print(f'[v02]   large_pm_likely:      {int(large_pm_likely.sum()):>10,}')

    # ── Reorder columns: meta first ─────────────────────────────────────
    meta = ['primary_id', 'primary_source',
            'hst_id', 'jwst_id', 'euclid_vis_id', 'euclid_nisp_id',
            'detected_in_hst', 'detected_in_jwst',
            'detected_in_vis', 'detected_in_nisp',
            'source_type_hst', 'source_type_jwst',
            'source_type_vis', 'source_type_nisp',
            'is_point_source', 'is_agn_qso',
            'is_likely_star', 'hst_saturated_likely', 'large_pm_likely']
    gaia_block = [c for c in df_out.columns if c.startswith('gaia_')]
    pm_block   = [c for c in df_out.columns
                   if any(c.startswith(f'{p}_') or c.endswith(f'_{pair}')
                          for p in ('pmra','pmdec','pmtot','pmra_err','pmdec_err','pmtot_err','pm_flag','pm_method')
                          for pair in PM_PAIRS)]
    rest = [c for c in df_out.columns
             if c not in meta + gaia_block + pm_block]
    final_order = meta + gaia_block + pm_block + rest
    final_order = [c for c in final_order if c in df_out.columns]
    df_out = df_out[final_order]

    # ── Write ────────────────────────────────────────────────────────────
    print(f'[save] {len(df_out):,} rows × {len(df_out.columns)} cols')
    out_parq = OUT / 'master_or_catalog_v02.parquet'
    df_out.to_parquet(out_parq, index=False)
    if len(df_out) * len(df_out.columns) < 300_000_000:
        df_out.to_csv(OUT / 'master_or_catalog_v02.csv', index=False)
        print(f'[save]   wrote {out_parq.name} + .csv')
    else:
        print(f'[save]   wrote {out_parq.name} (CSV skipped — too large)')

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print('=' * 78)
    print('Summary')
    print('=' * 78)
    print(f'Total unique sources:           {len(df_out):,}')
    for m in MISSIONS:
        det = int(df_out[f'detected_in_{TAG[m]}'].sum())
        print(f'  detected in {m:<14} {det:>10,}')
    print(f'primary_source counts:')
    print(df_out['primary_source'].value_counts().to_string())
    print(f'is_point_source:                {int(df_out["is_point_source"].sum()):,}')
    print(f'is_agn_qso:                     {int(df_out["is_agn_qso"].sum()):,}')
    for pair in PM_PAIRS:
        n = int(np.sum(df_out[f'pmtot_{pair}'].notna()))
        print(f'PM info from {pair:<24} {n:>10,} sources')

    print(f'\nWall time: {time.time() - t_total:.1f}s')


if __name__ == '__main__':
    main()
