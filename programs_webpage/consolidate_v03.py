"""Standalone consolidator for v03 inference partials.

Reads the per-survey checkpoint parquets in csvfiles_sn/_partial/, applies
the (fixed) coverage-aware 1-of-N rule + morphology + saturation + dN/dm
filters from infer_sn_v03.update_outputs, and rewrites
  csvfiles_sn/tbl_sn_candidates_v03.csv
  htmls/sn_search/v03/index.html

Use any time — does not interfere with the running inference workers.
"""
import sys, time, os, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
from infer_sn_v03 import update_outputs

CSV_DIR = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
LOOK    = CSV_DIR / "fits_lookup_v03.parquet"
MODELS  = CSV_DIR / "cnn_models_v03.pt"

def main():
    bundle = torch.load(MODELS, map_location="cpu", weights_only=False)
    thresholds = bundle["thresholds"]
    print(f"Thresholds: {thresholds}", flush=True)

    lk = pq.read_table(LOOK)
    pid = np.array(lk["primary_id"].to_pylist(), dtype=object)
    psrc = np.array(lk["primary_source"].to_pylist(), dtype=object)
    ra  = lk["ra"].to_numpy(zero_copy_only=False)
    dec = lk["dec"].to_numpy(zero_copy_only=False)
    th  = np.array(lk["tile_hst"].to_pylist(), dtype=object)
    tj  = np.array(lk["tile_jwst"].to_pylist(), dtype=object)
    te  = np.array(lk["tile_euclid"].to_pylist(), dtype=object)
    in_h = np.array(lk["in_hst"].to_pylist(), dtype=bool)
    in_j = np.array(lk["in_jwst"].to_pylist(), dtype=bool)
    in_e = np.array(lk["in_euclid"].to_pylist(), dtype=bool)
    N = len(pid)
    print(f"Consolidating {N:,} candidates ...", flush=True)

    final = "--final" in sys.argv
    update_outputs(N, pid, psrc, ra, dec, th, tj, te,
                   in_h, in_j, in_e, thresholds, final=final)
    print(f"Done. tbl_sn_candidates_v03.csv + webpage updated.", flush=True)


if __name__ == "__main__":
    main()
