"""Standalone watcher for v03 inference partials.

Reads the per-survey checkpoint parquets every WATCH_INTERVAL seconds
and rewrites the webpage + CSV via the FIXED update_outputs (no VIS
dependency). Run alongside the survey workers.

Usage:
    python watch_v03.py            # loop forever
    python watch_v03.py --once     # one-shot
"""
import sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
from infer_sn_v03 import update_outputs

CSV_DIR = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
LOOK    = CSV_DIR / "fits_lookup_v03.parquet"
MODELS  = CSV_DIR / "cnn_models_v03.pt"
INTERVAL = 15

def load_lookup():
    bundle = torch.load(MODELS, map_location="cpu", weights_only=False)
    thresholds = bundle["thresholds"]
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
    return (len(pid), pid, psrc, ra, dec, th, tj, te, in_h, in_j, in_e, thresholds)

def tick(state, final=False):
    N, pid, psrc, ra, dec, th, tj, te, in_h, in_j, in_e, thresholds = state
    update_outputs(N, pid, psrc, ra, dec, th, tj, te,
                   in_h, in_j, in_e, thresholds, final=final)

def main():
    once = "--once" in sys.argv
    final = "--final" in sys.argv
    state = load_lookup()
    print(f"[{time.strftime('%H:%M:%S')}] watching ... interval={INTERVAL}s", flush=True)
    while True:
        t0 = time.time()
        try:
            tick(state, final=final)
            print(f"[{time.strftime('%H:%M:%S')}] update_outputs {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR {e}", flush=True)
        if once:
            return
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
