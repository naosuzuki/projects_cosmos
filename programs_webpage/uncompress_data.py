"""Uncompress all .fits.gz files in the JWST and HST data directories.

Uses `gunzip -k` (keeps the .gz original) and runs 2 workers in parallel.
Skips files where the uncompressed sibling already exists, so this script
is safe to interrupt and re-run.

Run:
    python uncompress_data.py
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

JWST_DIR = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8")
HST_DIR  = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0")

MAX_WORKERS = 4     # disk I/O bottleneck; 4 saturates without thrash


def find_gz_files(dirs):
    """Find real .fits.gz files; skip macOS resource-fork '._*' siblings.
    Process smaller files first (within each dir, sorted by size asc) so
    HST (small) finishes fast and gives early progress."""
    out = []
    for d in dirs:
        files = []
        for f in d.glob("*.fits.gz"):
            if f.name.startswith("._"):
                continue
            files.append((f.stat().st_size, f))
        files.sort()  # smallest first
        out.extend(f for _, f in files)
    return out


def gunzip_keep(gz_path):
    """gunzip -k {gz_path}; skip if uncompressed sibling already exists."""
    out_path = gz_path.with_suffix("")     # strips .gz
    if out_path.exists():
        return ("SKIP", gz_path.name, 0.0, out_path.stat().st_size)
    t0 = time.time()
    try:
        subprocess.run(["gunzip", "-k", str(gz_path)], check=True,
                       capture_output=True)
    except subprocess.CalledProcessError as e:
        return ("ERR", gz_path.name, time.time() - t0, 0, e.stderr.decode())
    elapsed = time.time() - t0
    return ("OK", out_path.name, elapsed, out_path.stat().st_size)


def fmt_size(b):
    return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB"


def main():
    files = find_gz_files([HST_DIR, JWST_DIR])  # HST (small) first for fast wins
    total_gz = sum(f.stat().st_size for f in files)
    print(f"Files to process: {len(files)}", flush=True)
    print(f"  JWST: {len([f for f in files if 'JWST' in str(f)])}", flush=True)
    print(f"  HST:  {len([f for f in files if 'HST' in str(f)])}", flush=True)
    print(f"Total compressed size: {fmt_size(total_gz)}", flush=True)
    print(f"Estimated uncompressed: ~{fmt_size(total_gz * 3)}  (rough 3x ratio)", flush=True)
    print(f"Workers: {MAX_WORKERS}\n", flush=True)

    t_start = time.time()
    n_done = 0; n_ok = 0; n_skip = 0; n_err = 0; total_out_bytes = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(gunzip_keep, f) for f in files]
        for fut in as_completed(futs):
            res = fut.result()
            status = res[0]; name = res[1]; elapsed = res[2]
            n_done += 1
            if status == "OK":
                size = res[3]
                total_out_bytes += size
                n_ok += 1
                rate = size / elapsed / 1e6 if elapsed > 0 else 0
                print(f"  [{n_done:3d}/{len(files)}] OK   {name}   "
                      f"{fmt_size(size)}  {elapsed:5.1f}s  "
                      f"({rate:.0f} MB/s out)", flush=True)
            elif status == "SKIP":
                n_skip += 1
                print(f"  [{n_done:3d}/{len(files)}] SKIP {name}   "
                      f"(already exists)", flush=True)
            else:
                n_err += 1
                err_msg = res[4] if len(res) > 4 else ""
                print(f"  [{n_done:3d}/{len(files)}] ERR  {name}   "
                      f"{err_msg[:200]}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nFinished.  ok={n_ok}  skipped={n_skip}  errored={n_err}", flush=True)
    print(f"  Total new bytes: {fmt_size(total_out_bytes)}", flush=True)
    print(f"  Wall time: {elapsed_total/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
