#!/usr/bin/env python
"""
54_step5_cross_match.py — Step 5: cross-match the 4 Gaia-augmented catalogs
across 9 mission subsets.

For each subset (= "combination"), produce one INTERSECTION table:
each row is one unique star DETECTED BY EVERY LISTED MISSION.  Sources
not detected by all listed missions are dropped.

The 9 combinations:
  PAIRS (6):
    HST × JWST
    HST × Euclid_VIS
    HST × Euclid_NISP
    JWST × Euclid_VIS
    JWST × Euclid_NISP
    Euclid_VIS × Euclid_NISP
  TRIPLES (2):
    HST × JWST × Euclid_VIS
    HST × JWST × Euclid_NISP
  QUADRUPLE (1) — the master star catalog:
    HST × JWST × Euclid_VIS × Euclid_NISP

Match logic (per combination):
  Tier 1 (Gaia-anchored): rows sharing the same gaia_source_id are the
    SAME star — gold-standard tie.
  Tier 2 (positional): catalog-detection rows without Gaia in any mission
    are union-find clustered via search_around_sky between each mission
    pair with the pair-appropriate tolerance.
  Tier 3 (INTERSECTION filter): after clustering, KEEP ONLY clusters
    where every listed mission contributed a catalog detection
    (source_type ∈ {'catalog','catalog+gaia'}).  gaia_only_bright rows
    do NOT count as a detection for the intersection.

Output schema (one row per unique star, present in all listed missions):
  star_id              integer cluster id, renumbered 1..N
  tie_type             'gaia_anchored' / 'positional'
  detected_in_<m>      always True for every m in combination
  gaia_*               one canonical set of Gaia columns (24 cols), or NaN
  <orig_col>_<m>       all catalog/DAO columns from each mission, suffixed
                       by mission tag (hst, jwst, vis, nisp)

Parallelism: 9 ProcessPoolExecutor workers, one per combination.
"""
from __future__ import annotations
import os
import sys
import time
import warnings
from pathlib import Path
from itertools import combinations
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

# Short mission tags used to suffix columns
TAG = {
    'HST':         'hst',
    'JWST':        'jwst',
    'Euclid_VIS':  'vis',
    'Euclid_NISP': 'nisp',
}

# Per-pair positional matching tolerance (arcsec) for catalog-only rows.
# Gaia-anchored ties don't use this.  Baselines:
#   HST  ↔ JWST     19   yr
#   HST  ↔ Euclid   19.5 yr
#   JWST ↔ Euclid    0.5 yr
#   VIS  ↔ NISP      0   yr (same epoch)
PAIR_RADIUS_AS = {
    frozenset(['HST', 'JWST']):              0.50,
    frozenset(['HST', 'Euclid_VIS']):        0.50,
    frozenset(['HST', 'Euclid_NISP']):       0.60,
    frozenset(['JWST', 'Euclid_VIS']):       0.25,
    frozenset(['JWST', 'Euclid_NISP']):      0.30,
    frozenset(['Euclid_VIS', 'Euclid_NISP']): 0.30,
}

GAIA_COL_PREFIX = 'gaia_'   # columns we'll deduplicate across missions


# ─────────────────────────────────────────────────────────────────────────────
# Union-Find
# ─────────────────────────────────────────────────────────────────────────────

class UnionFind:
    __slots__ = ('parent', 'rank')

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank   = np.zeros(n, dtype=np.int32)

    def find(self, x: int) -> int:
        p = self.parent
        # iterative path compression
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            nxt = p[x]
            p[x] = root
            x = nxt
        return root

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def union_many(self, xs: np.ndarray, ys: np.ndarray) -> None:
        for x, y in zip(xs, ys):
            self.union(int(x), int(y))


# ─────────────────────────────────────────────────────────────────────────────
# Worker — cross-match one combination
# ─────────────────────────────────────────────────────────────────────────────

def cross_match_combo(missions: list[str], out_name: str) -> dict:
    """Cross-match the listed missions; write outer-join table."""
    t0 = time.time()
    tag = '+'.join(TAG[m] for m in missions)
    print(f'[{tag}] starting ({len(missions)} missions)')

    # ── Load augmented catalogs ────────────────────────────────────────────
    cats = {}
    offsets = {}        # offset of each mission in the global row index
    next_off = 0
    for m in missions:
        df = pd.read_parquet(OUT / f'cat_matched_{m}_with_gaia.parquet')
        cats[m] = df.reset_index(drop=True)
        offsets[m] = next_off
        next_off += len(df)
    n_total = next_off
    print(f'[{tag}]   total rows across missions: {n_total:,}')

    # Build per-mission boolean arrays we'll need
    # has_cat == True ⇔ source_type in {'catalog', 'catalog+gaia'} (cat_ra present)
    # has_gaia == True ⇔ gaia_source_id present (catalog+gaia OR gaia_only_bright)
    cat_ra_global  = np.full(n_total, np.nan)
    cat_dec_global = np.full(n_total, np.nan)
    gaia_id_global = np.full(n_total, -1, dtype=np.int64)
    mission_of     = np.empty(n_total, dtype=object)
    local_idx      = np.empty(n_total, dtype=np.int64)

    for m in missions:
        df = cats[m]
        off = offsets[m]
        n = len(df)
        cat_ra_global[off:off+n]  = df['cat_ra'].values
        cat_dec_global[off:off+n] = df['cat_dec'].values
        gid = df['gaia_source_id'].values
        gaia_id_global[off:off+n] = np.where(pd.isna(gid), -1, gid).astype(np.int64)
        mission_of[off:off+n] = m
        local_idx[off:off+n]  = np.arange(n, dtype=np.int64)

    has_cat  = ~np.isnan(cat_ra_global)
    has_gaia = gaia_id_global > 0

    # ── Union-find ─────────────────────────────────────────────────────────
    uf = UnionFind(n_total)

    # Tier 1: union by gaia_source_id (across missions and within mission)
    # Group rows by gaia_id and union them all to the first member.
    print(f'[{tag}]   tier 1 — Gaia-anchored union')
    gaia_present_idx = np.where(has_gaia)[0]
    if len(gaia_present_idx):
        gid_vals = gaia_id_global[gaia_present_idx]
        # Sort by gid → groups of equal gid are contiguous
        order = np.argsort(gid_vals, kind='stable')
        gaia_present_idx = gaia_present_idx[order]
        gid_vals = gid_vals[order]
        # For each run of equal gid, union all to the first
        run_starts = np.concatenate(([0], np.where(np.diff(gid_vals) != 0)[0] + 1))
        run_starts = np.append(run_starts, len(gid_vals))
        n_union_gaia = 0
        for k in range(len(run_starts) - 1):
            s, e = run_starts[k], run_starts[k+1]
            if e - s > 1:
                first = int(gaia_present_idx[s])
                for j in range(s+1, e):
                    uf.union(first, int(gaia_present_idx[j]))
                    n_union_gaia += 1
        print(f'[{tag}]     Gaia unions made: {n_union_gaia:,} '
              f'(across {len(run_starts)-1:,} unique Gaia stars)')

    # Tier 2: positional union for catalog-detected rows between each
    # mission pair.  We do NOT positional-match within a single mission.
    print(f'[{tag}]   tier 2 — positional union per mission pair')
    for m1, m2 in combinations(missions, 2):
        radius = PAIR_RADIUS_AS[frozenset([m1, m2])]
        # Subset: catalog-detected rows in each mission
        off1, off2 = offsets[m1], offsets[m2]
        n1, n2 = len(cats[m1]), len(cats[m2])
        sel1 = has_cat[off1:off1+n1]
        sel2 = has_cat[off2:off2+n2]
        idx1_local = np.where(sel1)[0]
        idx2_local = np.where(sel2)[0]
        if len(idx1_local) == 0 or len(idx2_local) == 0:
            continue
        c1 = SkyCoord(cat_ra_global[off1+idx1_local]  * u.deg,
                      cat_dec_global[off1+idx1_local] * u.deg)
        c2 = SkyCoord(cat_ra_global[off2+idx2_local]  * u.deg,
                      cat_dec_global[off2+idx2_local] * u.deg)
        p1, p2, sep, _ = search_around_sky(c1, c2, radius * u.arcsec)
        # Translate back to global indices
        g1 = off1 + idx1_local[p1]
        g2 = off2 + idx2_local[p2]
        # For each row in m1, keep only the closest m2 partner (1:1)
        if len(p1):
            order = np.argsort(sep.arcsec)
            p1s = p1[order]; p2s = p2[order]; sep_s = sep.arcsec[order]
            seen_a = np.zeros(len(idx1_local), dtype=bool)
            seen_b = np.zeros(len(idx2_local), dtype=bool)
            keep = np.zeros(len(p1s), dtype=bool)
            for i in range(len(p1s)):
                a, b = p1s[i], p2s[i]
                if not seen_a[a] and not seen_b[b]:
                    seen_a[a] = True
                    seen_b[b] = True
                    keep[i] = True
            ka = idx1_local[p1s[keep]]
            kb = idx2_local[p2s[keep]]
            uf.union_many(off1 + ka, off2 + kb)
            print(f'[{tag}]     {m1}↔{m2} at {radius}": '
                  f'{len(p1):,} candidate pairs → {keep.sum():,} unique 1:1 ties')

    # ── Build cluster table ────────────────────────────────────────────────
    print(f'[{tag}]   resolving cluster roots ...')
    roots = np.fromiter((uf.find(i) for i in range(n_total)),
                        dtype=np.int64, count=n_total)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    n_clusters = len(unique_roots)
    print(f'[{tag}]   {n_clusters:,} unique stars across the combination')

    # For each (cluster, mission), find one local row index that belongs.
    # When a cluster has multiple rows in the same mission (e.g. MER split
    # detections), we keep the FIRST one and add a *_n_detections counter.
    cluster_to_mission_row = {m: np.full(n_clusters, -1, dtype=np.int64)
                              for m in missions}
    n_detections_per_cluster = {m: np.zeros(n_clusters, dtype=np.int32)
                                for m in missions}

    for m in missions:
        off = offsets[m]
        n = len(cats[m])
        cl = inverse[off:off+n]   # cluster index for each local row
        # Count detections per cluster (only count catalog rows, not gaia_only_bright)
        is_cat_only_or_with_gaia = has_cat[off:off+n]
        np.add.at(n_detections_per_cluster[m], cl[is_cat_only_or_with_gaia], 1)
        # Pick first row per cluster — fill earliest occurrence
        # (numpy trick: iterate sorted-by-cluster; first wins)
        order = np.argsort(cl, kind='stable')
        cl_sorted = cl[order]
        # Mask of "first occurrence in cluster"
        first = np.concatenate(([True], cl_sorted[1:] != cl_sorted[:-1]))
        first_clusters = cl_sorted[first]
        first_local_rows = order[first]
        cluster_to_mission_row[m][first_clusters] = first_local_rows

    # Build output table
    out_cols = {}

    # Basic metadata
    out_cols['star_id'] = np.arange(1, n_clusters + 1, dtype=np.int64)

    # Per-mission detection flags & counts.  detected_in_<m> means "the
    # mission's photometric catalog DETECTED this star" — i.e. there is at
    # least one row of source_type ∈ {'catalog','catalog+gaia'} (has cat_ra).
    # A cluster whose only contribution from mission m is a 'gaia_only_bright'
    # row (Gaia-known, mission-saturated) has detected_in_<m>=False but
    # source_type_<m> will be 'gaia_only_bright' to indicate that.
    for m in missions:
        out_cols[f'detected_in_{TAG[m]}'] = n_detections_per_cluster[m] > 0
        out_cols[f'n_detections_{TAG[m]}'] = n_detections_per_cluster[m]

    # Canonical Gaia info: take from any mission row that has Gaia, prefer JWST > VIS > NISP > HST
    canonical_order = [m for m in ('JWST', 'Euclid_VIS', 'Euclid_NISP', 'HST') if m in missions]
    gaia_cols = [c for c in cats[missions[0]].columns if c.startswith(GAIA_COL_PREFIX)]
    # Pre-allocate
    for col in gaia_cols:
        # Use object dtype where source columns are non-numeric
        dt = cats[canonical_order[0]][col].dtype
        if dt.kind in ('f', 'i', 'u'):
            out_cols[col] = np.full(n_clusters, np.nan)
        else:
            out_cols[col] = np.full(n_clusters, None, dtype=object)
    has_gaia_cluster = np.zeros(n_clusters, dtype=bool)
    for m in canonical_order:
        idxs = cluster_to_mission_row[m]
        mask = (idxs >= 0) & ~has_gaia_cluster   # only fill clusters not yet filled
        if not mask.any():
            continue
        rows = idxs[mask]
        # Of those rows, which actually have Gaia data?
        gid_col = cats[m]['gaia_source_id'].values
        has_g = ~pd.isna(gid_col[rows])
        if not has_g.any():
            continue
        # Select clusters where this mission has Gaia
        sel_cluster = np.where(mask)[0][has_g]
        sel_local = rows[has_g]
        has_gaia_cluster[sel_cluster] = True
        for col in gaia_cols:
            vals = cats[m][col].values[sel_local]
            out_cols[col][sel_cluster] = vals

    # Per-mission catalog/DAO/source_type columns, suffixed by mission tag
    for m in missions:
        suffix = TAG[m]
        local_idxs = cluster_to_mission_row[m]
        mask = local_idxs >= 0
        for col in cats[m].columns:
            if col.startswith(GAIA_COL_PREFIX):
                continue   # handled above as canonical
            new_col = f'{col}_{suffix}'
            src = cats[m][col].values
            if src.dtype.kind in ('f', 'i', 'u'):
                arr = np.full(n_clusters, np.nan)
                arr[mask] = src[local_idxs[mask]]
            elif src.dtype.kind == 'b':
                arr = np.full(n_clusters, False)
                arr[mask] = src[local_idxs[mask]]
            else:
                arr = np.full(n_clusters, None, dtype=object)
                arr[mask] = src[local_idxs[mask]]
            out_cols[new_col] = arr

    # tie_type — in the intersection catalogs, every cluster has ≥1
    # mission detection, so 'gaia_only_bright' cannot survive.  Tag rows
    # with 'gaia_anchored' if any mission row in the cluster carries Gaia,
    # else 'positional'.
    tie_type = np.where(has_gaia_cluster, 'gaia_anchored', 'positional')
    out_cols['tie_type'] = tie_type

    out_df = pd.DataFrame(out_cols)

    # ── INTERSECTION filter ────────────────────────────────────────────────
    # Keep only clusters detected by EVERY listed mission (catalog detection).
    # gaia_only_bright clusters are excluded because they have no mission
    # photometric detection.
    det_cols = [f'detected_in_{TAG[m]}' for m in missions]
    intersect_mask = np.ones(len(out_df), dtype=bool)
    for c in det_cols:
        intersect_mask &= out_df[c].values
    n_before = len(out_df)
    out_df = out_df[intersect_mask].reset_index(drop=True)
    # Renumber star_id 1..N
    out_df['star_id'] = np.arange(1, len(out_df) + 1, dtype=np.int64)
    print(f'[{tag}]   intersection: {len(out_df):,} / {n_before:,} '
          f'clusters survive (detected by ALL {len(missions)} missions)')

    # Reorder columns: meta first, then gaia, then per-mission groups
    meta_cols  = ['star_id', 'tie_type'] + \
                 [f'detected_in_{TAG[m]}' for m in missions] + \
                 [f'n_detections_{TAG[m]}' for m in missions]
    meta_set = set(meta_cols)
    gaia_set = set(gaia_cols)
    final_order = meta_cols + gaia_cols
    for m in missions:
        suffix = TAG[m]
        final_order += [c for c in out_df.columns
                        if c.endswith(f'_{suffix}')
                        and c not in gaia_set
                        and c not in meta_set]
    # Sanity: no duplicates, all columns accounted for
    assert len(final_order) == len(set(final_order)), \
        f'duplicate columns in final_order: {[c for c in final_order if final_order.count(c)>1][:5]}'
    out_df = out_df[final_order]

    # ── Write ──────────────────────────────────────────────────────────────
    out_parq = OUT / f'cross_{out_name}.parquet'
    out_df.to_parquet(out_parq, index=False)
    # CSV only for outputs ≤ ~500 MB
    if len(out_df) * len(out_df.columns) < 200_000_000:   # ~200M cells limit
        out_csv = OUT / f'cross_{out_name}.csv'
        out_df.to_csv(out_csv, index=False)
        csv_written = True
    else:
        csv_written = False

    elapsed = time.time() - t0
    n_gaia    = int(np.sum(out_df['tie_type'] == 'gaia_anchored'))
    n_pos     = int(np.sum(out_df['tie_type'] == 'positional'))
    print(f'[{tag}]   wrote {out_parq.name}: {len(out_df):,} stars × {len(out_df.columns)} cols  '
          f'(gaia_anchored={n_gaia:,}, positional={n_pos:,}) '
          f'in {elapsed:.1f}s')

    # Per-mission count breakdown
    counts = {m: int(out_df[f'detected_in_{TAG[m]}'].sum()) for m in missions}

    return {
        'name': out_name,
        'missions': missions,
        'n_stars': int(len(out_df)),
        'n_union_before_filter': int(n_before),
        'n_gaia_anchored': n_gaia,
        'n_positional': n_pos,
        'mission_counts': counts,
        'elapsed_s': elapsed,
        'output_parquet': str(out_parq),
        'csv_written': csv_written,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main — dispatch 9 combinations in parallel
# ─────────────────────────────────────────────────────────────────────────────

COMBINATIONS = [
    # 6 pairs
    (['HST', 'JWST'],                                          'HST_JWST'),
    (['HST', 'Euclid_VIS'],                                    'HST_Euclid_VIS'),
    (['HST', 'Euclid_NISP'],                                   'HST_Euclid_NISP'),
    (['JWST', 'Euclid_VIS'],                                   'JWST_Euclid_VIS'),
    (['JWST', 'Euclid_NISP'],                                  'JWST_Euclid_NISP'),
    (['Euclid_VIS', 'Euclid_NISP'],                            'Euclid_VIS_Euclid_NISP'),
    # 2 triples
    (['HST', 'JWST', 'Euclid_VIS'],                            'HST_JWST_Euclid_VIS'),
    (['HST', 'JWST', 'Euclid_NISP'],                           'HST_JWST_Euclid_NISP'),
    # 1 four-way master
    (['HST', 'JWST', 'Euclid_VIS', 'Euclid_NISP'],
     'HST_JWST_Euclid_VIS_Euclid_NISP'),
]


def main():
    t_total = time.time()
    print('=' * 75)
    print('Step 5 — cross-match 9 mission combinations (parallel)')
    print('=' * 75)

    results = []
    with ProcessPoolExecutor(max_workers=9) as pool:
        futures = {pool.submit(cross_match_combo, missions, out_name):
                   out_name for missions, out_name in COMBINATIONS}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f'[{name}] FAILED: {e!r}')
                import traceback
                traceback.print_exc()
                raise

    # ── Summary table ──────────────────────────────────────────────────────
    print()
    print('=' * 90)
    print('Summary')
    print('=' * 90)
    lines = []
    lines.append(f'{"output":<40} {"k":>2} {"union_pre":>10} {"intersect":>10} '
                 f'{"gaia":>7} {"pos":>7} {"time":>7}')
    lines.append('-' * 90)
    for r in sorted(results, key=lambda x: (len(x['missions']), x['name'])):
        lines.append(
            f'{r["name"]:<40} {len(r["missions"]):>2} '
            f'{r["n_union_before_filter"]:>10,} '
            f'{r["n_stars"]:>10,} '
            f'{r["n_gaia_anchored"]:>7,} {r["n_positional"]:>7,} '
            f'{r["elapsed_s"]:>6.1f}s'
        )
    text = '\n'.join(lines)
    print(text)
    out_path = HTML / 'cross_match_summary.txt'
    out_path.write_text(text + '\n\n')

    # ── Per-combination per-mission detection counts ───────────────────────
    print()
    print('Per-mission detection counts:')
    miss_lines = []
    for r in sorted(results, key=lambda x: (len(x['missions']), x['name'])):
        line = f'  {r["name"]:<45} '
        for m, c in r['mission_counts'].items():
            line += f'{m}={c:,}  '
        miss_lines.append(line)
        print(line)
    out_path.write_text(text + '\n\n' + '\n'.join(miss_lines) + '\n')

    print(f'\nTotal wall time: {time.time() - t_total:.1f}s')
    print(f'Summary → {out_path}')


if __name__ == '__main__':
    main()
