"""Step 1 v03: build SN candidate source catalog from master v03.

Input:  csvfiles_star/master_or_catalog_v03.parquet  (1,215,146 rows × 303 cols)
Output: csvfiles_sn/sn_candidates_v03.parquet        (surviving rows, same schema)
        csvfiles_sn/sn_candidates_v03_summary.txt    (exclusion audit log)

Filter (vastly simpler than v01 — the master v03 has already aggregated
all stellar evidence into a single boolean column):

  EXCLUDE if  is_likely_star == True
  EXCLUDE if  is_agn_qso     == True

Master v03's `is_likely_star` aggregates Gaia source_id, cat_class_star_hst,
cat_phz_classification (1=STAR per verified encoding), cat_point_like_prob,
hst_saturated_likely (B3 workaround), and pm_significant (5σ + 5 mas/yr in
any cross-mission PM pair).

`is_agn_qso` comes from COSMOS-Web LePhare type==2 OR flag_chandra>0.5.

Validated upstream: 0/15 of the known SNe present in master are flagged
as is_likely_star.

The mag>22 filter is applied DOWNSTREAM at inference time (per-band, based
on detection survey), not here — so the candidate catalog retains full
context for any later analysis or threshold tuning.
"""
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

IN_PATH  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_star/master_or_catalog_v03.parquet")
OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
OUT_PARQ = OUT_DIR / "sn_candidates_v03.parquet"
OUT_LOG  = OUT_DIR / "sn_candidates_v03_summary.txt"


def main():
    t0 = time.time()
    print(f"Reading {IN_PATH.name} ...", flush=True)
    table = pq.read_table(IN_PATH)
    n_total = table.num_rows
    print(f"  {n_total:,} rows × {table.num_columns} columns", flush=True)

    is_star = np.asarray(table["is_likely_star"].to_numpy(zero_copy_only=False), dtype=bool)
    is_agn  = np.asarray(table["is_agn_qso"].to_numpy(zero_copy_only=False), dtype=bool)
    n_star = int(is_star.sum())
    n_agn  = int(is_agn.sum())
    n_both = int((is_star & is_agn).sum())

    exclude = is_star | is_agn
    keep = ~exclude
    n_keep = int(keep.sum())
    n_excl = int(exclude.sum())

    print(f"Filtering: keep {n_keep:,} of {n_total:,} ({100*n_keep/n_total:.2f}%) ...", flush=True)
    out = table.filter(pa.array(keep))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing {OUT_PARQ} ...", flush=True)
    pq.write_table(out, OUT_PARQ, compression="zstd")
    size_mb = OUT_PARQ.stat().st_size / 1e6
    print(f"  wrote {out.num_rows:,} rows × {out.num_columns} cols  ({size_mb:.1f} MB)", flush=True)

    log = [
        "# SN candidate source catalog v03 — build summary",
        f"# Built: {time.strftime('%Y-%m-%d %H:%M:%S')}  ({time.time()-t0:.1f} sec total)",
        f"# Source: {IN_PATH}",
        f"# Output: {OUT_PARQ}",
        "",
        "## Exclusion rules",
        "EXCLUDE if  is_likely_star == True   (master v03 aggregation)",
        "EXCLUDE if  is_agn_qso     == True",
        "",
        "## Row counts",
        f"Total in master v03:            {n_total:>10,}",
        f"  Excluded:                     {n_excl:>10,}  ({100*n_excl/n_total:.2f}%)",
        f"  Kept (SN candidates):         {n_keep:>10,}  ({100*n_keep/n_total:.2f}%)",
        "",
        "## Per-rule exclusion counts (with overlap)",
        f"is_likely_star:                 {n_star:>10,}",
        f"is_agn_qso:                     {n_agn:>10,}",
        f"  ∩ (both True):                {n_both:>10,}",
        "",
        f"## File size: {size_mb:.1f} MB (zstd parquet)",
    ]
    OUT_LOG.write_text("\n".join(log) + "\n")
    print(f"Wrote {OUT_LOG}", flush=True)


if __name__ == "__main__":
    main()
