"""Build the MASTER supernova list = the 17 user-known SNe + the validated
subset of the 65-SN DeCoursey COSMOS-Web catalog (sn_ori.txt).

Workflow:
  1. This script seeds  csvfiles_sn/master_sn_verdicts.csv  with all 17 known
     (status=real, locked) + 65 catalog candidates (status=pending). The user
     edits the `status` column to 'real' or 'bogus' after visual inspection
     of htmls/sn_search/catalog65_inspect/index.html.
  2. Re-run with --compile to emit  csvfiles_sn/master_sn_list.csv  containing
     every status=='real' SN with full tile assignments, ready as the v06
     positive truth set.

Usage:
    python build_master_sn.py            # seed the verdicts file (idempotent:
                                         # preserves any verdicts already set)
    python build_master_sn.py --compile  # write master_sn_list.csv from verdicts
"""
import sys, csv
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

CSV_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
KNOWN17  = CSV_DIR / "lookup_sn17_v32.csv"
CAT65    = CSV_DIR / "sn_catalog65_inspect.csv"
VERDICTS = CSV_DIR / "master_sn_verdicts.csv"
MASTER   = CSV_DIR / "master_sn_list.csv"
LOOK     = CSV_DIR / "fits_lookup_v03.parquet"

COLS = ["sn_name","source","status","telescope","sn_ra","sn_dec",
        "hst_tile","jwst_tile","euclid_tile","z","disc_filter","notes"]


def tiles_for(ra, dec, lk):
    cosd = np.cos(np.deg2rad(dec))
    sep = np.sqrt(((lk["ra"].values-ra)*cosd*3600)**2 + ((lk["dec"].values-dec)*3600)**2)
    j = int(np.argmin(sep))
    if sep[j] < 2.0:
        return (str(lk["tile_hst"].iloc[j]), str(lk["tile_jwst"].iloc[j]),
                str(lk["tile_euclid"].iloc[j]))
    return ("","","")


def seed():
    lk = pq.read_table(LOOK).to_pandas()
    # preserve existing verdicts if file already exists
    prior = {}
    if VERDICTS.exists():
        for r in csv.DictReader(VERDICTS.open()):
            prior[r["sn_name"]] = r.get("status",""), r.get("notes","")

    rows = []
    # --- 17 known (always real, locked) ---
    for r in csv.DictReader(KNOWN17.open()):
        rows.append(dict(sn_name=r["id"], source="known17", status="real",
                         telescope=r["telescope"], sn_ra=r["sn_ra"], sn_dec=r["sn_dec"],
                         hst_tile=r.get("hst",""), jwst_tile=r.get("jwst",""),
                         euclid_tile=r.get("euclid",""), z="", disc_filter="",
                         notes="user-confirmed known SN"))
    # --- 65 catalog candidates (pending until user marks) ---
    for r in csv.DictReader(CAT65.open()):
        st, nt = prior.get(r["id"], ("pending",""))
        # never downgrade a known; catalog names are disjoint from the 17 (verified)
        cov = "" if r["hst_tile"] else "NO_FITS_COVERAGE"
        rows.append(dict(sn_name=r["id"], source="cat65", status=st or "pending",
                         telescope=r.get("telescope","JWST"),
                         sn_ra=r["sn_ra"], sn_dec=r["sn_dec"],
                         hst_tile=r["hst_tile"], jwst_tile=r["jwst_tile"],
                         euclid_tile=r["euclid_tile"], z=r.get("z",""),
                         disc_filter=r.get("filt",""), notes=nt or cov))
    with VERDICTS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow(r)
    n_pend = sum(1 for r in rows if r["status"]=="pending")
    print(f"seeded {VERDICTS}")
    print(f"  17 known (locked real) + 65 catalog candidates ({n_pend} pending)")
    print(f"  → edit the 'status' column to 'real' or 'bogus', then run --compile")


def compile_master():
    if not VERDICTS.exists():
        sys.exit("run without --compile first to seed the verdicts file")
    rows = [r for r in csv.DictReader(VERDICTS.open()) if r["status"]=="real"]
    with MASTER.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow(r)
    from collections import Counter
    bs = Counter(r["source"] for r in rows)
    tel = Counter(r["telescope"] for r in rows)
    print(f"wrote {MASTER}: {len(rows)} real SNe  (sources {dict(bs)}; telescopes {dict(tel)})")
    cov = sum(1 for r in rows if r["jwst_tile"])
    print(f"  with FITS coverage (trainable): {cov}")


if __name__ == "__main__":
    if "--compile" in sys.argv: compile_master()
    else: seed()
