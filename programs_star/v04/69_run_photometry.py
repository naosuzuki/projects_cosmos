#!/usr/bin/env python
"""
69_run_photometry.py — resumable, timed driver for the Step 3a-③ + 3b
chain over all detected tiles:  53_ (forced PSF photometry + saturation
+ DAO morphology) → 67_ (3b star classifier) → 66_ (DAO PSF-fitting
photometry on the point-source pool).

Mirrors 65_run_chi2_detection.py: parallel across tiles, per-mission
concurrency tuned to the single external HDD, idempotent per stage
(each stage's meta JSON is the resume marker), per-tile wall time
appended to a git-tracked CSV.

Only tiles with a completed 51_ detection meta enter the worklist.

Usage:
  ./69_run_photometry.py                          # jwst,hst,euclid
  ./69_run_photometry.py --missions euclid --jobs 4
  ./69_run_photometry.py --stages 53,67           # skip 66_
  ./69_run_photometry.py --tiles A4 --force
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

TIMING_DIR = Path('/Users/suzuki/github/projects_cosmos/programs_star/csv_timing')
TIMING_DIR.mkdir(parents=True, exist_ok=True)
TIMING_CSV = TIMING_DIR / 'phot_timing.csv'
TIMING_COLS = ['utc', 'mission', 'tile', 'stage', 'status', 'secs',
               'n_rows', 'n_is_star', 'n_pool']

DEFAULT_JOBS = {'jwst': 2, 'hst': 3, 'euclid': 3}

STAGES = {
    '53': dict(script='53_step3a_dual_photometry.py',
               meta='phot/phot_{t}.meta.json'),
    '67': dict(script='67_step3b_classifier.py',
               meta='phot/star_{t}.meta.json'),
    '66': dict(script='66_step3a_dao_psf_photometry.py',
               meta='phot/dao_{t}.meta.json'),
}
STAGE_ORDER = ['53', '67', '66']

_lock = threading.Lock()


def detected_tiles(mission: str) -> list[str]:
    return sorted(p.parent.name for p in
                  WORK.glob(f'{mission}_chi2/*/detect_*.meta.json'))


def append_timing(row):
    with _lock:
        new = not TIMING_CSV.exists()
        with open(TIMING_CSV, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=TIMING_COLS)
            if new:
                w.writeheader()
            w.writerow(row)


def run_tile(mission: str, tile: str, stages: list[str], force: bool):
    tdir = WORK / f'{mission}_chi2' / tile
    results = []
    for st in stages:
        meta = tdir / STAGES[st]['meta'].format(t=tile)
        if meta.exists() and not force:
            results.append((st, 'skip', 0))
            continue
        cmd = [PY, str(HERE / STAGES[st]['script']),
               '--mission', mission, '--tile', tile] + \
              (['--force'] if force else [])
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        secs = round(time.time() - t0, 1)
        ok = r.returncode == 0 and meta.exists()
        if not ok:
            log = tdir / f'_phot_{st}_{tile}.fail.log'
            log.write_text(r.stdout + '\n--- STDERR ---\n' + r.stderr)
        m = {}
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
            except Exception:
                pass
        append_timing({
            'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'mission': mission, 'tile': tile, 'stage': st,
            'status': 'ok' if ok else 'fail', 'secs': secs,
            'n_rows': m.get('n_rows', m.get('n_sources', '')),
            'n_is_star': m.get('n_is_star', ''),
            'n_pool': m.get('n_pool_any', ''),
        })
        results.append((st, 'ok' if ok else 'fail', secs))
        if not ok:
            break                     # later stages depend on this one
    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--missions', default='jwst,hst,euclid')
    p.add_argument('--stages', default='53,67,66',
                   help='comma subset of 53,67,66 (order fixed)')
    p.add_argument('--jobs', type=int, default=None)
    p.add_argument('--tiles', default='')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    missions = [m.strip() for m in args.missions.split(',') if m.strip()]
    want = {s.strip() for s in args.stages.split(',') if s.strip()}
    stages = [s for s in STAGE_ORDER if s in want]
    keep = {t.strip() for t in args.tiles.split(',') if t.strip()}
    print(f'photometry chain: missions={missions} stages={stages}')
    print(f'timing CSV → {TIMING_CSV}')
    g0 = time.time()
    for mission in missions:
        tiles = detected_tiles(mission)
        if keep:
            tiles = [t for t in tiles if t in keep]
        jobs = args.jobs or DEFAULT_JOBS[mission]
        print(f'\n=== {mission.upper()}: {len(tiles)} tiles, '
              f'{jobs}-way ===', flush=True)
        ok = fail = 0
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_tile, mission, t, stages, args.force): t
                    for t in tiles}
            for i, fut in enumerate(as_completed(futs), 1):
                tile = futs[fut]
                res = fut.result()
                bad = [s for s, st, _ in res if st == 'fail']
                tag = ' '.join(f'{s}:{st}{"" if st=="skip" else f"({sec:.0f}s)"}'
                               for s, st, sec in res)
                if bad:
                    fail += 1
                else:
                    ok += 1
                print(f'  [{i}/{len(tiles)}] {"FAIL" if bad else "ok  "} '
                      f'{mission}/{tile:12s} {tag}', flush=True)
        print(f'=== {mission.upper()}: {ok} ok, {fail} fail ===', flush=True)
    print(f'\n######## wall {time.time()-g0:.0f}s ########')


if __name__ == '__main__':
    main()
