#!/usr/bin/env python
"""
65_run_chi2_detection.py — resumable, timed driver for Step 3a-② χ²₊
hot+cold detection (51_step3a_chi2_detect.py) across missions.

Mirrors 60_run_mass_production.py: parallel across tiles, per-mission
concurrency tuned to the single external HDD, idempotent (a tile whose
detect meta exists is skipped), per-tile wall-time appended to a
git-tracked CSV as it finishes.

Memory note: a JWST tile holds the χ²₊ accumulator plus one band's
SCI/WHT at a time (≈6 GB peak on the big A10 tile), so JWST runs
2-way; HST is one band (3-way); Euclid frames are 10200² (4-way).

Usage:
  ./65_run_chi2_detection.py                          # jwst,hst,euclid
  ./65_run_chi2_detection.py --missions jwst --tiles A4
  ./65_run_chi2_detection.py --force
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY   = '/opt/miniconda3/bin/python'
WORK = Path('/Volumes/exdisk1/data/photometry_v04')
EUC  = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')

TIMING_DIR = Path('/Users/suzuki/github/projects_cosmos/programs_star/csv_timing')
TIMING_DIR.mkdir(parents=True, exist_ok=True)
TIMING_CSV = TIMING_DIR / 'chi2_timing.csv'
TIMING_COLS = ['utc', 'mission', 'tile', 'status', 'stack_s', 'cold_s',
               'hot_s', 'merge_s', 'total_s', 'n_cold', 'n_hot_kept',
               'n_total']

AB_TILES = [f'{r}{n}' for r in 'AB' for n in range(1, 11)]
DEFAULT_JOBS = {'jwst': 2, 'hst': 3, 'euclid': 4}

_lock = threading.Lock()


def euclid_tiles():
    return sorted({p.name.split('TILE')[1].split('-')[0]
                   for p in EUC.glob('EUC_MER_BGSUB-MOSAIC-VIS_TILE*.fits')})


def build_worklist(missions):
    items = []
    for m in missions:
        tiles = AB_TILES if m in ('jwst', 'hst') else euclid_tiles()
        for t in tiles:
            items.append(dict(
                mission=m, tile=t,
                meta=WORK / f'{m}_chi2' / t / f'detect_{t}.meta.json',
                cmd=[PY, str(HERE / '51_step3a_chi2_detect.py'),
                     '--mission', m, '--tile', t]))
    return items


def append_timing(row):
    with _lock:
        new = not TIMING_CSV.exists()
        with open(TIMING_CSV, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=TIMING_COLS)
            if new:
                w.writeheader()
            w.writerow(row)


def run_item(it, force):
    tag = f"{it['mission']}/{it['tile']}"
    if it['meta'].exists() and not force:
        return ('skip', tag, 0)
    cmd = it['cmd'] + (['--force'] if force else [])
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    secs = round(time.time() - t0, 1)
    status = 'ok' if (r.returncode == 0 and it['meta'].exists()) else 'fail'
    if status == 'fail':
        log = it['meta'].parent / f"_detect_{it['tile']}.fail.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(r.stdout + '\n--- STDERR ---\n' + r.stderr)
    m = {}
    if it['meta'].exists():
        try:
            m = json.loads(it['meta'].read_text())
        except Exception:
            pass
    append_timing({
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'mission': it['mission'], 'tile': it['tile'], 'status': status,
        'stack_s': m.get('stack_s', ''), 'cold_s': m.get('cold_s', ''),
        'hot_s': m.get('hot_s', ''), 'merge_s': m.get('merge_s', ''),
        'total_s': m.get('total_s', secs),
        'n_cold': m.get('n_cold', ''),
        'n_hot_kept': m.get('n_hot_kept', ''),
        'n_total': m.get('n_total', ''),
    })
    return (status, tag, secs)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--missions', default='jwst,hst,euclid')
    p.add_argument('--jobs', type=int, default=None,
                   help='override per-mission concurrency')
    p.add_argument('--force', action='store_true')
    p.add_argument('--tiles', default='',
                   help='comma list of tiles to keep; others dropped')
    return p.parse_args()


def main():
    args = parse_args()
    missions = [m.strip() for m in args.missions.split(',') if m.strip()]
    items = build_worklist(missions)
    keep = {t.strip() for t in args.tiles.split(',') if t.strip()}
    if keep:
        items = [it for it in items if it['tile'] in keep]
        print(f'tiles kept: {sorted(keep)}')
    print(f'chi2 detection: missions={missions}  items={len(items)}')
    print(f'timing CSV → {TIMING_CSV}')
    g0 = time.time()
    grand_ok = grand_fail = 0
    for mission in missions:
        mitems = [it for it in items if it['mission'] == mission]
        todo = [it for it in mitems if args.force or not it['meta'].exists()]
        jobs = args.jobs or DEFAULT_JOBS[mission]
        print(f'\n=== {mission.upper()}: {len(mitems)} tiles '
              f'({len(mitems)-len(todo)} done, {len(todo)} to run, '
              f'{jobs}-way) ===', flush=True)
        ok = fail = 0
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_item, it, args.force): it for it in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                status, tag, secs = fut.result()
                if status == 'ok':
                    ok += 1
                elif status == 'fail':
                    fail += 1
                mark = 'OK ' if status == 'ok' else status.upper()
                print(f'  [{i}/{len(todo)}] {mark:5s} {tag:24s} '
                      f'{secs:7.0f}s', flush=True)
        print(f'=== {mission.upper()}: {ok} ok, {fail} fail ===', flush=True)
        grand_ok += ok
        grand_fail += fail
    print(f'\n######## GRAND TOTAL: {grand_ok} ok, {grand_fail} fail, '
          f'wall {time.time()-g0:.0f}s ########')


if __name__ == '__main__':
    main()
