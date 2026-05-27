"""Step 1 of SN hunt: build SN candidate source catalog.

Input:  csvfiles_star/master_or_catalog_v01.parquet  (1,132,959 rows × 292 cols)
Output: csvfiles_sn/sn_candidates_v01.parquet        (surviving rows, same schema)
        csvfiles_sn/sn_candidates_v01_summary.txt    (exclusion audit log)

Exclusion rule: drop a row from the master catalog if it is a known star
(via PM evidence) or a known AGN/QSO.

  EX1  is_agn_qso == True
  EX2  Gaia DR3 PM measured: gaia_pmra and gaia_pmdec both finite.
       (Gaia in COSMOS detects essentially only stars; measured PM is the
        cleanest single-source confirmation.)
  EX3  pm_flag_<pair> == "gaia_consistent" for ANY of the 5 cross-mission
       pairs. This is a cross-mission PM that has been validated against
       Gaia, so it is a Gaia-anchored star observed independently.

We intentionally DO NOT exclude on pm_flag == "ok": probe of the master
shows 107K of 208K HST×Euclid_VIS rows would pass |pmtot|>max(5,3σ) with
flag=ok, which is implausible for stars and consistent with astrometric
residuals across a 19.5-yr baseline. Without Gaia validation the
cross-mission PM signal is not a reliable star indicator.

We also intentionally DO NOT exclude on is_point_source: SNe are point
sources by definition.
"""
import sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

IN_PATH  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_star/master_or_catalog_v01.parquet")
OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
OUT_PARQ = OUT_DIR / "sn_candidates_v01.parquet"
OUT_LOG  = OUT_DIR / "sn_candidates_v01_summary.txt"

PAIRS = ["HST_Euclid_VIS", "HST_Euclid_NISP", "HST_JWST",
         "JWST_Euclid_VIS", "JWST_Euclid_NISP"]


def main():
    t0 = time.time()
    print(f"Reading {IN_PATH.name} ...", flush=True)
    table = pq.read_table(IN_PATH)
    n_total = table.num_rows
    print(f"  {n_total:,} rows × {table.num_columns} columns", flush=True)

    # --- EX1: AGN/QSO ---------------------------------------------------
    is_agn = np.asarray(table["is_agn_qso"].to_numpy(zero_copy_only=False), dtype=bool)
    n_agn = int(is_agn.sum())

    # --- EX2: Gaia DR3 PM available -------------------------------------
    pmra  = table["gaia_pmra"].to_numpy(zero_copy_only=False)
    pmdec = table["gaia_pmdec"].to_numpy(zero_copy_only=False)
    gaia_pm_star = np.isfinite(pmra) & np.isfinite(pmdec)
    n_gaia = int(gaia_pm_star.sum())

    # --- EX3: any cross-mission pair has pm_flag == "gaia_consistent" ---
    pair_gaia_consistent = np.zeros(n_total, dtype=bool)
    pair_counts = {}
    for pp in PAIRS:
        flags = table[f"pm_flag_{pp}"].to_pylist()
        m = np.array([f == "gaia_consistent" for f in flags], dtype=bool)
        pair_gaia_consistent |= m
        pair_counts[pp] = int(m.sum())
    n_xmission_gaia = int(pair_gaia_consistent.sum())

    # --- combined exclusion mask ----------------------------------------
    exclude = is_agn | gaia_pm_star | pair_gaia_consistent
    keep = ~exclude
    n_keep = int(keep.sum())
    n_excl = int(exclude.sum())

    # overlaps among the three exclusion sets (for audit)
    n_agn_only       = int((is_agn & ~gaia_pm_star & ~pair_gaia_consistent).sum())
    n_gaia_only      = int((~is_agn & gaia_pm_star & ~pair_gaia_consistent).sum())
    n_xmgaia_only    = int((~is_agn & ~gaia_pm_star & pair_gaia_consistent).sum())
    n_agn_and_gaia   = int((is_agn & gaia_pm_star).sum())
    n_agn_and_xm     = int((is_agn & pair_gaia_consistent).sum())
    n_gaia_and_xm    = int((gaia_pm_star & pair_gaia_consistent).sum())
    n_all_three      = int((is_agn & gaia_pm_star & pair_gaia_consistent).sum())

    # --- apply mask and write -------------------------------------------
    print(f"Filtering: keep {n_keep:,} of {n_total:,} ({100*n_keep/n_total:.2f}%) ...", flush=True)
    mask = pa.array(keep)
    out = table.filter(mask)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing {OUT_PARQ} ...", flush=True)
    pq.write_table(out, OUT_PARQ, compression="zstd")
    out_size_mb = OUT_PARQ.stat().st_size / 1e6
    print(f"  wrote {out.num_rows:,} rows × {out.num_columns} cols  ({out_size_mb:.1f} MB)", flush=True)

    # --- audit log ------------------------------------------------------
    log = []
    log.append("# SN candidate source catalog v01 — build summary")
    log.append(f"# Built: {time.strftime('%Y-%m-%d %H:%M:%S')}  ({time.time()-t0:.1f} sec total)")
    log.append(f"# Source: {IN_PATH}")
    log.append(f"# Output: {OUT_PARQ}")
    log.append("")
    log.append("## Exclusion rules")
    log.append("EX1: is_agn_qso == True")
    log.append("EX2: gaia_pmra and gaia_pmdec both finite (Gaia DR3 PM)")
    log.append('EX3: any of 5 cross-mission pm_flag_* == "gaia_consistent"')
    log.append("")
    log.append("## Row counts")
    log.append(f"Total in master:                {n_total:>10,}")
    log.append(f"  Excluded:                     {n_excl:>10,}  ({100*n_excl/n_total:.2f}%)")
    log.append(f"  Kept (SN candidates):         {n_keep:>10,}  ({100*n_keep/n_total:.2f}%)")
    log.append("")
    log.append("## Per-rule exclusion counts")
    log.append(f"EX1 is_agn_qso True:            {n_agn:>10,}")
    log.append(f"EX2 Gaia PM star:               {n_gaia:>10,}")
    log.append(f"EX3 xmission gaia_consistent:   {n_xmission_gaia:>10,}")
    log.append("")
    log.append("## Per-pair gaia_consistent counts (component of EX3)")
    for pp, c in pair_counts.items():
        log.append(f"  {pp:<22s}  {c:>10,}")
    log.append("")
    log.append("## Overlap audit (Venn breakdown)")
    log.append(f"  EX1 only:                     {n_agn_only:>10,}")
    log.append(f"  EX2 only:                     {n_gaia_only:>10,}")
    log.append(f"  EX3 only:                     {n_xmgaia_only:>10,}")
    log.append(f"  EX1 ∩ EX2:                    {n_agn_and_gaia:>10,}")
    log.append(f"  EX1 ∩ EX3:                    {n_agn_and_xm:>10,}")
    log.append(f"  EX2 ∩ EX3:                    {n_gaia_and_xm:>10,}")
    log.append(f"  EX1 ∩ EX2 ∩ EX3:              {n_all_three:>10,}")
    log.append("")
    log.append("## Schema")
    log.append(f"Columns: {out.num_columns}  (same as input)")
    log.append("File size: %.1f MB  (zstd parquet)" % out_size_mb)

    OUT_LOG.write_text("\n".join(log) + "\n")
    print(f"Wrote {OUT_LOG}", flush=True)
    print("\n--- audit log ---")
    print("\n".join(log))


if __name__ == "__main__":
    main()
